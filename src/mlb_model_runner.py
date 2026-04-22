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

    # Use UTC time to avoid GitHub Actions date issues
    today = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

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

    # Features and target
    features = ['hard_hit_rate', 'launch_speed_avg', 'launch_angle_avg']
    X = train_df[features]
    y = train_df['is_hr']

    print("🤖 Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)

    # Get latest stats per batter
    print("🔮 Generating predictions...")
    latest = train_df.sort_values('game_date').groupby('batter').last().reset_index()
    latest_X = latest[features]
    latest['hr_prob'] = model.predict_proba(latest_X)[:, 1]

    # Look up player names
    print("👤 Looking up player names...")
    batter_ids = latest['batter'].dropna().astype(int).unique().tolist()
    try:
        id_lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
        id_lookup['full_name'] = id_lookup['name_first'].str.capitalize() + ' ' + id_lookup['name_last'].str.capitalize()
        id_map = dict(zip(id_lookup['key_mlbam'], id_lookup['full_name']))
    except Exception as e:
        print(f"⚠️ Player lookup failed: {e}")
        id_map = {}

    # Build predictions list
    predictions = []
    for _, row in latest.iterrows():
        prob = round(float(row['hr_prob']) * 100, 4)
        batter_id = int(row['batter'])
        player_name = id_map.get(batter_id, 'Unknown Player')

        edge = round((prob / 100) - 0.5, 4)

        if prob >= 2:
            confidence = "High"
        elif prob >= 1:
            confidence = "Medium"
        else:
            confidence = "Low"

        predictions.append({
            "game_date": today,
            "player_name": player_name,
            "team": "TBD",
            "prop_type": "Home Run",
            "projected_prob": prob,
            "implied_line": 0.5,
            "edge": edge,
            "confidence": confidence
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

    # Filter out unknown players from top picks
    high_conf_named = [p for p in high_conf if p['player_name'] != 'Unknown Player']
    medium_conf_named = [p for p in medium_conf if p['player_name'] != 'Unknown Player']
    low_conf_named = [p for p in low_conf if p['player_name'] != 'Unknown Player']

    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
    embed = DiscordEmbed(
        title="⚾ MLB Prop Model Daily Update",
        description=f"Generated **{len(predictions)}** predictions for **{today}**",
        color="03b2f8"
    )
    embed.add_embed_field(name="🔥 High Confidence (≥2%)", value=str(len(high_conf)), inline=True)
    embed.add_embed_field(name="✅ Medium Confidence (≥1%)", value=str(len(medium_conf)), inline=True)
    embed.add_embed_field(name="📉 Low Confidence (<1%)", value=str(len(low_conf)), inline=True)

    if high_conf_named:
        top_picks = sorted(high_conf_named, key=lambda x: x['projected_prob'], reverse=True)[:5]
        top_str = "\n".join([
            f"**{p['player_name']}** — {p['projected_prob']:.2f}% HR prob"
            for p in top_picks
        ])
        embed.add_embed_field(name="🎯 Top Picks Today", value=top_str, inline=False)
    elif medium_conf_named:
        top_picks = sorted(medium_conf_named, key=lambda x: x['projected_prob'], reverse=True)[:5]
        top_str = "\n".join([
            f"**{p['player_name']}** — {p['projected_prob']:.2f}% HR prob"
            for p in top_picks
        ])
        embed.add_embed_field(name="🎯 Top Picks (Medium)", value=top_str, inline=False)
    elif low_conf_named:
        top_picks = sorted(low_conf_named, key=lambda x: x['projected_prob'], reverse=True)[:5]
        top_str = "\n".join([
            f"**{p['player_name']}** — {p['projected_prob']:.2f}% HR prob"
            for p in top_picks
        ])
        embed.add_embed_field(name="🎯 Top Picks (Low)", value=top_str, inline=False)

    webhook.add_embed(embed)
    webhook.execute()
    print("📣 Discord notification sent!")

if __name__ == "__main__":
    run_mlb_model()
