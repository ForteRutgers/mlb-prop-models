import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
import pybaseball as pyb
import xgboost as xgb
from discord_webhook import DiscordWebhook, DiscordEmbed
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

# ==========================================
# ==== CONFIG FROM .env ====
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not all([SUPABASE_URL, SUPABASE_KEY, DISCORD_WEBHOOK_URL]):
    raise ValueError("Missing environment variables in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def run_mlb_model():
    print("🚀 Starting MLB Prop Model Runner...")

    # Step 1: Pull Statcast data
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

    print(f"📊 Pulling Statcast data from {start_date} to {today}...")
    statcast_df = pyb.statcast(start_dt=start_date, end_dt=today)

    if statcast_df.empty:
        print("⚠️ No data pulled. Exiting.")
        return

    # Basic cleaning
    statcast_df = statcast_df.dropna(subset=['game_date', 'batter'])
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date']).dt.date
    statcast_df['is_hr'] = (statcast_df['events'] == 'home_run').astype(int)

    # Feature engineering
    print("🔧 Engineering features...")
    df = statcast_df.copy()

    # Create a "hard hit" column (launch speed >= 95 mph) to replace the missing barrel column
    # We add .fillna(0) to safely handle pitches that were not hit (like strikeouts or walks)
    df['hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)

    df['hard_hit_rate'] = df.groupby('batter')['hard_hit'].transform(lambda x: x.rolling(50, min_periods=10).mean())
    df['launch_speed_avg'] = df.groupby('batter')['launch_speed'].transform(
        lambda x: x.rolling(30, min_periods=10).mean())
    df['launch_angle_avg'] = df.groupby('batter')['launch_angle'].transform(
        lambda x: x.rolling(30, min_periods=10).mean())

    train_df = df.dropna(subset=['hard_hit_rate', 'launch_speed_avg', 'launch_angle_avg'])

    # Train XGBoost model (real machine learning)
    print("🧠 Training XGBoost model...")
    features = ['hard_hit_rate', 'launch_speed_avg', 'launch_angle_avg']
    X = train_df[features]
    y = train_df['is_hr']

    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X, y)

    # Generate today's predictions
    print("🔮 Generating today's predictions...")
    recent = df[df['game_date'] >= (datetime.today() - timedelta(days=30)).date()]

    player_stats = recent.groupby('batter').agg({
        'hard_hit_rate': 'mean',
        'launch_speed_avg': 'mean',
        'launch_angle_avg': 'mean'
    }).reset_index()

    # Placeholder names (we'll improve this later)
    player_stats['player_name'] = "Player_" + player_stats['batter'].astype(str)
    player_stats['team'] = "TBD"

    player_stats['projected_prob'] = model.predict_proba(player_stats[features])[:, 1] * 100

    # Prepare predictions for Supabase
    predictions = []
    for index, row in player_stats.iterrows():
        if row['projected_prob'] > 0:
            predictions.append({
                "game_date": today,
                "player_name": row['player_name'],
                "team": row['team'],
                "prop_type": "Home Run",
                "projected_prob": float(row['projected_prob']),
                "implied_line": 0.5,
                "edge": float(row['projected_prob'] - 50),
                "confidence": "High" if row['projected_prob'] >= 70 else "Medium"
            })

    # Insert into Supabase
    if predictions:
        print(f"💾 Inserting {len(predictions)} predictions into Supabase...")
        supabase.table("prop_predictions").insert(predictions).execute()
        print("✅ Predictions saved successfully!")
    else:
        print("⚠️ No predictions met threshold.")

    # Send Discord notification
    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
    embed = DiscordEmbed(
        title="MLB Prop Model Update",
        description=f"Generated {len(predictions)} predictions for {today}",
        color="00FF00"
    )
    embed.add_embed_field(
        name="High Confidence (>=70%)",
        value=str(len([p for p in predictions if p.get('projected_prob', 0) >= 70])),
        inline=True
    )
    webhook.add_embed(embed)
    webhook.execute()
    print("📣 Discord notification sent!")


if __name__ == "__main__":
    run_mlb_model()