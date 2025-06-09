import pandas as pd

def build_weekly_normalized_table(file_path_or_df, min_posts=1):
    """
    Load a hashtag dataset and return a weekly normalized engagement table.

    Parameters:
    - file_path_or_df: str or pd.DataFrame — path to CSV file or a preloaded DataFrame
    - min_posts: int — minimum number of posts per hashtag per week to be included

    Returns:
    - pd.DataFrame with columns:
        ['hashtags_name', 'year', 'week', 'weekly_engagement',
         'post_count', 'normalized_weekly_engagement', 'date']
    """

    # Load CSV if a path is passed
    if isinstance(file_path_or_df, str):
        df = pd.read_csv(file_path_or_df)
    else:
        df = file_path_or_df.copy()

    # Rename column if needed
    if 'hashtag_name' in df.columns and 'hashtags_name' not in df.columns:
        df = df.rename(columns={"hashtag_name": "hashtags_name"})

    # Drop rows without required fields
    required_columns = ['hashtags_name', 'createTimeISO', 'diggCount', 'shareCount', 'playCount', 'collectCount', 'commentCount', 'id']
    df = df.dropna(subset=[col for col in required_columns if col in df.columns])

    # Convert createTimeISO to datetime
    df['createTimeISO'] = pd.to_datetime(df['createTimeISO'], errors='coerce')

    # Drop rows with invalid dates
    df = df[df['createTimeISO'].notna()]

    # Compute engagement
    df['engagement'] = (
        df.get('diggCount', 0) +
        df.get('shareCount', 0) +
        df.get('playCount', 0) +
        df.get('collectCount', 0) +
        df.get('commentCount', 0)
    )

    # Extract year and week
    df['year'] = df['createTimeISO'].dt.year
    df['week'] = df['createTimeISO'].dt.isocalendar().week

    # Group and aggregate
    weekly_stats = (
        df.groupby(['hashtags_name', 'year', 'week'])
        .agg(
            weekly_engagement=('engagement', 'sum'),
            post_count=('post_id', 'nunique')
        )
        .reset_index()
    )

    

    # Apply rolling average smoothing (window=2 or 3 usually works well)
    weekly_stats['smoothed_post_count'] = (
        weekly_stats.groupby('hashtags_name')['post_count']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )


    # Filter by minimum post count
    weekly_stats = weekly_stats[weekly_stats['post_count'] >= min_posts]

    # Normalize engagement within each week
    weekly_stats['normalized_weekly_engagement'] = (
        weekly_stats.groupby(['year', 'week'])['weekly_engagement']
        .transform(lambda x: x / x.max())
    )

    # Add date column for time-series plotting (Monday of the week)
    weekly_stats['date'] = pd.to_datetime(
        weekly_stats['year'].astype(str) + '-' + weekly_stats['week'].astype(str) + '-1',
        format='%Y-%W-%w'
    )

    weekly_stats = weekly_stats.sort_values(['hashtags_name', 'date'])

    # Apply rolling average smoothing for past three weeks 
    weekly_stats['smoothed_post_count'] = (
        weekly_stats.groupby('hashtags_name')['post_count']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    
    return weekly_stats.sort_values(['year', 'week', 'normalized_weekly_engagement'], ascending=[True, True, False])
