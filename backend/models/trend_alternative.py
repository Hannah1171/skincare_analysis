import burst_detection
import numpy as np
import pandas as pd
from backend.models.topics import generate_topic_label, get_filtered_keywords, run_topic_model
from backend.models.trends import preprocess_posts
from backend.preprocessing.preprocessing import get_stopwords
from scipy.stats import median_abs_deviation


def get_trends(df: pd.DataFrame, min_volume: int = 5, top_n: int = 10):
    # Preprocess raw posts
    df = df[df["textLanguage"] == "en"].copy()
    df = preprocess_posts(df)

    # Run topic modeling directly inside trends pipeline
    model, topic_summary, df_named = run_topic_model(df, min_cluster_size=5, min_samples=2, text_col='text')

    # Assign topics to dataframe
    df_named["topic"] = df_named["Topic"]  # Ensure uniform column name

    # Filter out noise
    df_named = df_named[df_named["topic"] != -1].copy()

    # Weekly share change
    weekly_changes = compute_weekly_share_change(df_named)

    # Trend detection: count frequencies per day
    daily_counts = (
        df_named.groupby([df_named["date"].dt.date, "topic"])
        .size()
        .reset_index(name="Frequency")
        .rename(columns={"date": "Timestamp", "topic": "Topic"})
    )

    # Filter low-volume topics
    topic_totals = daily_counts.groupby("Topic")["Frequency"].sum()
    daily_counts = daily_counts[daily_counts["Topic"].isin(topic_totals[topic_totals >= min_volume].index)]

    # Detect trend signals
    z_scores = detect_zscore_trends(daily_counts)
    burst_scores = detect_burst_trends(daily_counts, df_named)

    # Combine scores
    combined_scores = {
        tid: max(z_scores.get(tid, 0), burst_scores.get(tid, 0))
        for tid in set(z_scores) | set(burst_scores)
    }

    print(combined_scores)
    print(topic_summary)

    # Label trends
    top_trends_df = label_trends(combined_scores, weekly_changes, model, df_named, top_n=top_n)

    return top_trends_df, df_named, daily_counts


def compute_weekly_share_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    weekly_counts = df.groupby(['week', 'topic']).size().reset_index(name='count')
    total_counts = df.groupby('week').size().reset_index(name='total')

    weekly_counts = weekly_counts.merge(total_counts, on='week')
    weekly_counts['share'] = weekly_counts['count'] / weekly_counts['total']

    weekly_counts.sort_values(['topic', 'week'], inplace=True)
    weekly_counts['prev_share'] = weekly_counts.groupby('topic')['share'].shift(1)
    weekly_counts['weekly_share_change'] = weekly_counts['share'] - weekly_counts['prev_share']

    latest_week = weekly_counts['week'].max()

    return (
        weekly_counts[weekly_counts['week'] == latest_week][['topic', 'weekly_share_change']]
        .rename(columns={'topic': 'TopicID'})
        .fillna(0)
    )
 
def detect_zscore_trends(trend_df: pd.DataFrame, z_thresh=3.5, smooth_win=3, min_history=2) -> dict[int, float]:
    df = trend_df.copy()
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    total_per_day = df.groupby("day")["Frequency"].sum().to_dict()

    scores = {}
    for topic_id, grp in df.groupby("Topic"):
        if topic_id == -1 or len(grp) < min_history:
            continue

        grp = grp.sort_values("day")
        dates = grp["day"].to_numpy()
        freqs = grp["Frequency"].astype(float).to_numpy()

        valid_mask = [total_per_day.get(d, 0) > 0 for d in dates]
        if sum(valid_mask) < min_history:
            continue

        dates = dates[valid_mask]
        freqs = freqs[valid_mask]
        ratios = np.array([freq / total_per_day[d] for freq, d in zip(freqs, dates)])

        smoothed = pd.Series(ratios, index=dates).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
        hist = smoothed[:-1]
        med = np.median(hist)
        mad = median_abs_deviation(hist, scale="normal") or 1
        z = (smoothed[-1] - med) / mad

        if z > z_thresh:
            scores[topic_id] = float(z)

    return scores

def detect_burst_trends(trend_df: pd.DataFrame, posts_df: pd.DataFrame, min_history=2) -> dict[int, int]:
    daily_posts = posts_df.assign(day=pd.to_datetime(posts_df['date']).dt.normalize()).groupby('day').size()

    df = trend_df.copy()
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()

    burst_scores = {}
    for topic_id, grp in df.groupby("Topic"):
        if topic_id == -1 or len(grp) < min_history:
            continue

        grp = grp.sort_values("day")
        freqs = grp["Frequency"].astype(float).to_numpy()
        days = grp["day"]
        ds = daily_posts.reindex(days, fill_value=0).to_numpy()

        valid = (ds > 0) & np.isfinite(freqs)
        r, d = freqs[valid], ds[valid]
        if len(r) < min_history or r.sum() == 0 or d.sum() == 0:
            continue

        try:
            q, *_ = burst_detection(r, d, len(r), s=2.0, gamma=1.0, smooth_win=1)
            if q and q[-1] > 0:
                burst_scores[topic_id] = int(q[-1])
        except Exception:
            continue

    return burst_scores

def label_trends(trend_scores, weekly_changes, model, df, top_n=10):
    stopwords = get_stopwords(langs=["en"])
    common = set(stopwords) | {"skin", "product", "care", "use"}

    top_items = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for topic_id, score in top_items:
        kws = get_filtered_keywords(model, topic_id, common, top_n=5)
        examples = df.loc[df['topic'] == topic_id, 'text'].dropna().head(3).tolist()
        label = generate_topic_label(kws, examples)

        weekly_change = weekly_changes.loc[weekly_changes['TopicID'] == topic_id, 'weekly_share_change']
        weekly_change_value = weekly_change.values[0] if not weekly_change.empty else 0.0

        rows.append({
            "TopicID": topic_id,
            "Label": label,
            "Trend_Score": round(score, 2),
            "Weekly_Share_Change": round(weekly_change_value, 2),
            "Keywords": kws
        })

    return pd.DataFrame(rows)
