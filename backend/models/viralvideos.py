import pandas as pd
from datetime import datetime, timedelta

def get_top_viral_videos(file_path: str):
    # Load CSV data into DataFrame
    df = pd.read_csv(file_path)

    # Parse 'date' column as datetime, setting invalid dates as NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows where 'date' could not be parsed
    df = df.dropna(subset=['date'])

    # Create a 'week' column based on ISO week number (1–52)
    df['week'] = df['date'].dt.isocalendar().week

    # Remove exact duplicate videos based on video URL (or video ID if available)
    df = df.drop_duplicates(subset='webVideoUrl')

    # Keep only videos from the last 7 days
    last_7_days = datetime.utcnow().date() - timedelta(days=7)
    df_recent = df[df['date'].dt.date >= last_7_days]

    # Sort recent videos by play count (highest first), remove duplicates again, and take top 5
    top5_weekly = (
        df_recent.sort_values('playCount', ascending=False)
                 .drop_duplicates('webVideoUrl')  # Safety check for duplicates
                 .head(5)                         # Take top 5
                 .reset_index(drop=True)          # Clean up index
    )

    # Return the final DataFrame with top 5 viral videos
    return top5_weekly


