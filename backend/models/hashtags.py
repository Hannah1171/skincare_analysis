import pandas as pd
from datetime import timedelta

def get_weekly_hashtag_trends(df):

    if isinstance(df, str):
        df = pd.read_csv(df, parse_dates=['createTimeISO'])
    else:
        df = df.copy()
    # Preprocess
    df['post_date'] = df['createTimeISO'].dt.date
    df['hashtag_name'] = df['hashtag_name'].str.lower().str.strip()

    # Date ranges
    today = df['post_date'].max()
    this_week_start = today - timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)
    week_before_last_start = last_week_start - timedelta(days=7)

    # Weekly slices
    df_this_week = df[df['post_date'] >= this_week_start]
    df_last_week = df[(df['post_date'] >= last_week_start) & (df['post_date'] < this_week_start)]
    df_week_before_last = df[(df['post_date'] >= week_before_last_start) & (df['post_date'] < last_week_start)]

    # Helper function
    def get_hashtag_share(df_week):
        unique = df_week[['post_id', 'hashtag_name']].drop_duplicates()
        counts = unique.groupby('hashtag_name').size().reset_index(name='mentions')
        total_posts = unique['post_id'].nunique()
        counts['share'] = counts['mentions'] / total_posts
        return counts.set_index('hashtag_name')[['share']]

    # Weekly shares
    this_week_share = get_hashtag_share(df_this_week).rename(columns={'share': 'this_week_share'})
    last_week_share = get_hashtag_share(df_last_week).rename(columns={'share': 'last_week_share'})
    week_before_last_share = get_hashtag_share(df_week_before_last).rename(columns={'share': 'week_before_last_share'})

    # Merge
    trend_df = this_week_share.join(last_week_share, how='outer').join(week_before_last_share, how='outer').fillna(0)

    # Relative changes
    trend_df['trend_change_pct'] = ((trend_df['this_week_share'] - trend_df['last_week_share']) / trend_df['last_week_share'].replace(0, 1)).round(2)
    trend_df['last_week_trend_pct'] = ((trend_df['last_week_share'] - trend_df['week_before_last_share']) / trend_df['week_before_last_share'].replace(0, 1)).round(2)

    # IQR thresholds
    q1 = trend_df['trend_change_pct'].quantile(0.25)
    q3 = trend_df['trend_change_pct'].quantile(0.75)
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    lower_threshold = q1 - 1.5 * iqr

    def label_change(pct):
        if pct > upper_threshold:
            return "📈 Rising"
        elif pct < lower_threshold:
            return "📉 Dropping"
        else:
            return "⏸️ Stable"

    trend_df['trend_label'] = trend_df['trend_change_pct'].apply(label_change)
    trend_df['last_week_trend_label'] = trend_df['last_week_trend_pct'].apply(label_change)

    # Top hashtags
    top_tags = trend_df.sort_values('this_week_share', ascending=False).head(20).index.tolist()

    # Example posts
    example_posts = (
        df_this_week[df_this_week['hashtag_name'].isin(top_tags)]
        .sort_values('diggCount', ascending=False)
        .groupby('hashtag_name')
        .agg({
            'text': 'first',
            'webVideoUrl': 'first',
            'diggCount': 'first',
            'post_date': 'first'
        })
    )

    # Final merge
    final_df = example_posts.join(trend_df, how='left').sort_values('this_week_share', ascending=False)
    final_df_display = final_df.reset_index()[[
        'hashtag_name',
        'post_date',
        'diggCount',
        'this_week_share',
        'last_week_share',
        'week_before_last_share',
        'trend_change_pct',
        'trend_label',
        'last_week_trend_pct',
        'last_week_trend_label',
        'text',
        'webVideoUrl'
    ]]

    return final_df_display


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

    # Count total unique posts per week (across all hashtags)
    weekly_post_totals = (
    df.groupby(['year', 'week'])['post_id']
    .nunique()
    .reset_index()
    .rename(columns={'post_id': 'total_weekly_posts'})
    )

   # Merge into weekly_stats
    weekly_stats = pd.merge(
        weekly_stats,
        weekly_post_totals,
        on=['year', 'week'],
        how='left'
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
    weekly_stats['post_share_of_week'] = (
    weekly_stats['post_count'] / weekly_stats['total_weekly_posts']
    )

    return weekly_stats.sort_values(['year', 'week', 'normalized_weekly_engagement'], ascending=[True, True, False])
