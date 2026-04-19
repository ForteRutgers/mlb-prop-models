import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
import pybaseball as pyb
from pybaseball import playerid_reverse_lookup
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

    try:
        pyb.cache.enable()
    except:
        pass

    statcast_df = pyb.statcast(start_dt=start_date, end_dt=today)

    if statcast_df is None or statcast_df.empty:
        print("⚠️ No data pulled. Exiting.")
        return

    # Basic cleaning
    statcast_df = statcast_df.dropna(subset=['game_date', 'batter'])
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date']).dt.date
    statcast_df['is_hr'] = (statcast_df['events'] == 'home_run').astype(int)

    # Feature engineering
    print("🔧 Engineering features...")
    df = statcast_df.copy()

    df['hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)
    df['hard_hit_rate'] = df.groupby('batter')['hard_hit'].transform(
        lambda x: x.rolling(50, min_periods=10).mean()
    )
    df['launch_speed_avg'] = df.groupby('batter')['launch_speed'].transform(
        lambda x: x.rolling(30, min_periods=10).mean()
    )
    df['launch_angle_avg'] = df.groupby('batter')['launch_angle'].transform(
        lambda x: x.rolling(30, min_periods=10).mean()
    )

    train_df = df.dropna(subset=['hard_hit_rate', 'launch_speed_avg', 'launch_angle_avg'])

    if train_df.empty:
        print("⚠️ Training data is empty after feature engineering. Exiting.")
        return

    # Train XGBoost model
    print("🧠 Training XGBoost model...")
    features = ['hard_hit_rate', 'launch_speed_avg', 'launch_angle_avg']
    X = train_df[features]
    y = train_df['is_hr']

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X, y)

    # Generate today's predictions
    print("🔮 Generating today's predictions...")
    recent = df[df['game_date'] >= (datetime.today() - timedelta(days=30)).date()]

    player_stats = recent.groupby('batter').agg({
        'hard_hit_rate': 'mean',
        'launch_speed_avg': 'mean',
        'launch_angle_avg': 'mean'
    }).reset_index()

    player_stats = player_stats.dropna(subset=features)

    if player_stats.empty:
        print("⚠️ No player stats available. Exiting.")
        return

    # Look up real player names
    print("👤 Looking up player names...")
    try:
        batter_ids = player_stats['batter'].tolist()
        id_map = playerid_reverse_lookup(batter_ids, key_type='mlbam')
        id_map['full_name'] = id_map['name_first'].str.capitalize() + ' ' + id_map['name_last'].str.capitalize()
        id_map = id_map[['key_mlbam', 'full_name']]
        player_stats = player_stats.merge(id_map, left_on='batter', right_on='key_mlbam', how='left')
        player_stats['player_name'] = player_stats['full_name'].fillna(
            "Player_" + player_stats['batter'].astype(str)
        )
    except Exception as e:
        print(f"⚠️ Name lookup failed: {e} — using player IDs instead")
        player_stats['player_name'] = "Player_" + player_stats['batter'].astype(str)

    player_stats['team'] = "TBD"
    player_stats['projected_prob'] = model.predict_proba(player_stats[features])[:, 1] * 100

    # Prepare predictions for Supabase
    predictions = []
    for index, row in player_stats.iterrows():
        prob = float(row['projected_prob'])
        edge = round((prob / 100) - 0.5, 4)
        predictions.append({
            "game_date": today,
            "player_name": row['player_name'],
            "team": row['team'],
            "prop_type": "Home Run",
            "projected_prob": round(prob, 4),
            "implied_line": 0.5,
            "edge": edge,
            "confidence": "High" if prob >= 70 else "Medium" if prob >= 55 else "Low"
        })

    # Insert into Supabase
    if predictions:
        print(f"💾 Inserting {len(predictions)} predictions into Supabase...")
        supabase.table("prop_predictions").insert(predictions).execute()
        print("✅ Predictions saved successfully!")
    else:
        print("⚠️ No predictions generated.")

    # Send Discord notification
    high_conf = [p for p in predictions if p['confidence'] == 'High']
    medium_conf = [p for p in predictions if p['confidence'] == 'Medium']
    low_conf = [p for p in predictions if p['confidence'] == 'Low']

    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
    embed = DiscordEmbed(
        title="⚾ MLB Prop Model Daily Update",
        description=f"Generated **{len(predictions)}** predictions for **{today}**",
        color="03b2f8"
    )
    embed.add_embed_field(name="🔥 High Confidence (≥70%)", value=str(len(high_conf)), inline=True)
    embed.add_embed_field(name="✅ Medium Confidence (≥55%)", value=str(len(medium_conf)), inline=True)
    embed.add_embed_field(name="📉 Low Confidence (<55%)", value=str(len(low_conf)), inline=True)

    if high_conf:
        top_picks = sorted(high_conf, key=lambda x: x['projected_prob'], reverse=True)[:5]
        top_str = "\n".join([
            f"**{p['player_name']}** — {p['projected_prob']:.1f}% HR prob"
            for p in top_picks
        ])
        embed.add_embed_field(name="🎯 Top Picks Today", value=top_str, inline=False)

    webhook.add_embed(embed)
    webhook.execute()
    print("📣 Discord notification sent!")


if __name__ == "__main__":
    run_mlb_model()
