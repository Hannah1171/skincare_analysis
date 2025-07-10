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
