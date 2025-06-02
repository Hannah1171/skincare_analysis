import pandas as pd

def get_top_viral_videos(file_path: str):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.to_period('M')

    #df['engagement'] = df['diggCount'] + df['shareCount'] + df['commentCount'] + df['collectCount']

    top5_weekly = (
        df.sort_values(['week', 'playCount'], ascending=[True, False])
          .groupby('week')
          .head(5)
          .reset_index(drop=True)
    )

    top5_monthly = (
        df.sort_values(['month', 'playCount'], ascending=[True, False])
          .groupby('month')
          .head(5)
          .reset_index(drop=True)
    )

    return top5_weekly, top5_monthly
