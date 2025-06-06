# Hashtags frequenccy and playcount on weekly, combined hashtags
import pandas as pd

def build_weekly_normalized_table(hashtag_posts_filtered, min_posts=1):
    # Step 1: Aggregate engagement per hashtag per week
    weekly_stats = (
        hashtag_posts_filtered
        .groupby(['hashtags_name', 'year', 'week'])
        .agg(weekly_engagement=('engagement', 'sum'),
             post_count=('id', 'nunique'))
        .reset_index()
    )

    # Step 2: Filter hashtags that have enough posts in that week
    weekly_stats = weekly_stats[weekly_stats['post_count'] >= min_posts]

    # Step 3: Normalize weekly engagement within each week
    weekly_stats['normalized_weekly_engagement'] = (
        weekly_stats.groupby(['year', 'week'])['weekly_engagement']
        .transform(lambda x: x / x.max())
    )

    # Step 4: Add a date column for time series plotting
    weekly_stats['date'] = pd.to_datetime(
        weekly_stats['year'].astype(str) + '-' + weekly_stats['week'].astype(str) + '-1',
        format='%Y-%W-%w'
    )

    return weekly_stats.sort_values(['year', 'week', 'normalized_weekly_engagement'], ascending=[True, True, False])