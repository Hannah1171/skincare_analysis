from datetime import timedelta
import pandas as pd

def get_top5_trending_music(df):

    # Accept either a CSV filepath or a pre-loaded DataFrame
    if isinstance(df, str):
        df = pd.read_csv(df, parse_dates=['createTimeISO'])
    else:
        df = df.copy()

    # Exclude "original sounds" (user-generated audio)
    df = df[df['musicOriginal'] != True]

    # Extract posting date and ensure music IDs are strings
    df['post_date'] = pd.to_datetime(df['createTimeISO']).dt.date
    df['musicID'] = df['musicID'].astype(str)

    # Define this week (last 7 days) and previous week
    today = df['post_date'].max()
    this_week_start = today - timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)

    # Filter posts by date ranges
    df_this_week = df[df['post_date'] >= this_week_start]
    df_last_week = df[(df['post_date'] >= last_week_start) & (df['post_date'] < this_week_start)]

    # Calculate each track's share of total posts in a given week
    def get_music_share(df_week):
        unique = df_week[['post_id', 'musicID']].drop_duplicates()  # Avoid double-counting post-music pairs
        counts = unique.groupby('musicID').size().reset_index(name='mentions')
        total_posts = unique['post_id'].nunique()
        counts['share'] = counts['mentions'] / total_posts
        return counts.set_index('musicID')[['share']]

    # Get music shares for both weeks
    this_week_share = get_music_share(df_this_week).rename(columns={'share': 'this_week_share'})
    last_week_share = get_music_share(df_last_week).rename(columns={'share': 'last_week_share'})

    # Merge weekly shares and compute trend change percentage
    trend_df = this_week_share.join(last_week_share, how='outer').fillna(0)
    trend_df['trend_change_pct'] = (
        (trend_df['this_week_share'] - trend_df['last_week_share']) / trend_df['last_week_share'].replace(0, 1)
    ).round(2)

    # Identify top 15 tracks based on this week's share
    top_music = trend_df.sort_values('this_week_share', ascending=False).head(15).index.tolist()

    # Select example metadata for each top track
    example_posts = (
        df_this_week[df_this_week['musicID'].isin(top_music)]
        .sort_values('post_date', ascending=False)
        .groupby('musicID')
        .agg({
            'musicName': 'first',
            'musicAuthor': 'first',
            'playUrl': 'first',
            'post_date': 'first'
        })
    )

    # Combine example post details with trend data
    final_df = example_posts.join(trend_df, how='left').reset_index()

    # Return final ranked DataFrame
    return final_df.sort_values('this_week_share', ascending=False)
