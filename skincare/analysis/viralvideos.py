import pandas as pd

def get_top_viral_videos(file_path: str):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df['week'] = df['date'].dt.isocalendar().week

    # Drop exact video duplicates based on unique ID or URL
    df = df.drop_duplicates(subset='webVideoUrl')  # or use 'video_id' if you have it

    top5_weekly = (
        df.sort_values(['week', 'playCount'], ascending=[True, False])
          .groupby('week')
          .head(5)
          .reset_index(drop=True)
    )

    return top5_weekly
