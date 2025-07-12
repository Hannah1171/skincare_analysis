import pandas as pd
from datetime import datetime, timedelta


def get_top_viral_videos(file_path: str):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df['week'] = df['date'].dt.isocalendar().week

    # Drop exact video duplicates based on unique ID or URL
    df = df.drop_duplicates(subset='webVideoUrl')  # or use 'video_id' if you have it
    # Limit to last 7 days only
    last_7_days = datetime.utcnow().date() - timedelta(days=7)
    df_recent = df[df['date'].dt.date >= last_7_days]

    top5_weekly = (
      df_recent.sort_values('playCount', ascending=False)
             .drop_duplicates('webVideoUrl')
             .head(5)
             .reset_index(drop=True)
    )

    return top5_weekly



