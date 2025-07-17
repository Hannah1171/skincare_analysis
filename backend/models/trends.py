import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from burst_detection import burst_detection
from backend.preprocessing.preprocessing import get_stopwords, clean_text
from backend.models.topics import (
    build_topic_model,
    get_filtered_keywords,
    generate_topic_label,
    embedding_model
)

# patch
if not hasattr(np, 'float'):
    np.float = float


def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input dataframe for topic modeling.
    - Converts timestamps to datetime.
    - Combines 'text' and 'transcribed_text' into a unified document.
    - Cleans text by removing stopwords and applying general preprocessing.
    """
    stopwords = get_stopwords(langs=["en"])
    df = df.copy()
    df['date'] = pd.to_datetime(df['createTimeISO'])
    df['doc'] = (
        df['text'].fillna('') + ' ' + df['transcribed_text'].fillna('')
    ).str.strip().apply(lambda t: clean_text(t, stopwords=stopwords))
    return df


def train_topic_model(docs: list[str]):
    """
    Trains a topic model using the preprocessed documents.
    Returns:
        model: trained topic model object
        topics: topic assignment for each document
    """
    model = build_topic_model(
        min_cluster_size=11,
        min_samples=6,
        embedding_model=embedding_model,
        ngram_range=(1, 2)
    )
    topics, _ = model.fit_transform(documents=docs)
    return model, topics


def detect_trends(
    trend_df: pd.DataFrame,
    z_thresh: float = 3.5,
    smooth_win: int = 3,
    min_history: int = 4
) -> dict[int, float]:
    """
    Detect topics with emerging trends based on robust z-scores.
    - Normalizes topic frequencies per day.
    - Applies rolling mean smoothing.
    - Computes z-score of latest day's frequency vs. historical median.
    Returns:
        scores: Dictionary mapping topic IDs to their z-scores (only above threshold).
    """
    df = trend_df.copy()
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    total_per_day = df.groupby("day")["Frequency"].sum().to_dict()

    scores = {}
    for topic_id, grp in df.groupby("Topic"):
        if topic_id == -1 or len(grp) < min_history:
            continue  # Skip noise topics or insufficient history

        grp = grp.sort_values("day")
        dates = grp["day"].to_numpy()
        freqs = grp["Frequency"].astype(float).to_numpy()

        # Exclude days with no posts
        valid_mask = [total_per_day.get(d, 0) > 0 for d in dates]
        if sum(valid_mask) < min_history:
            continue

        dates = dates[valid_mask]
        freqs = freqs[valid_mask]

        # Normalize topic frequency by total posts for that day
        ratios = np.array([freq / total_per_day[d] for freq, d in zip(freqs, dates)])

        # Smooth series to reduce short-term fluctuations
        series = pd.Series(ratios, index=dates)
        smoothed = series.rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()

        # Compute robust z-score comparing last day's frequency to historical median
        hist = smoothed[:-1]  # Exclude latest day from history
        med = np.median(hist)
        mad = median_abs_deviation(hist, scale="normal") or 1  # Protect against division by zero
        z = (smoothed[-1] - med) / mad

        if z > z_thresh:
            scores[topic_id] = float(z)

    return scores


def detect_bursts(
    trend_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    s: float = 2.0,
    gamma: float = 1.0,
    min_history: int = 4
) -> dict[int, int]:
    """
    Detect sudden bursts in topic frequencies using Kleinberg's algorithm.
    - Aligns topic counts with total post counts per day.
    - Uses Kleinberg's burst detection to quantify sudden activity spikes.
    Returns:
        burst_scores: Dictionary mapping topic IDs to their burst level.
    """
    # Compute total posts per day as denominator for normalization
    daily_posts = (
        posts_df
        .assign(day=lambda df: pd.to_datetime(df['createTimeISO']).dt.normalize())
        .groupby('day')
        .size()
    )

    df = trend_df.copy()
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    df = df.sort_values("Timestamp")

    burst_scores = {}
    for topic_id, grp in df.groupby("Topic"):
        if topic_id == -1 or len(grp) < min_history:
            continue  # Skip noise topics or insufficient data

        grp = grp.sort_values("day")
        freqs = grp["Frequency"] \
            .apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x) \
            .astype(float).to_numpy()

        days = grp["day"]
        ds = daily_posts.reindex(days, fill_value=0).to_numpy()

        # Filter out invalid days (no posts or non-finite frequencies)
        valid = (ds > 0) & np.isfinite(freqs)
        r, d = freqs[valid], ds[valid]
        if len(r) < min_history or r.sum() == 0 or d.sum() == 0:
            continue

        # Apply Kleinberg burst detection algorithm
        try:
            q, _, _, _ = burst_detection(r, d, len(r), s=s, gamma=gamma, smooth_win=1)
            if q and q[-1] > 0:
                burst_scores[topic_id] = int(q[-1])  # Keep latest burst level if active
        except Exception:
            continue  # Robust against failures in burst detection

    return burst_scores


def label_top_trends(
    trend_scores: dict[int, float],
    model,
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generate descriptive labels for top trending topics:
    - Uses topic keywords and example posts.
    - Adds recent 2-week trend change data.
    Returns:
        DataFrame summarizing top topics with their label, trend score, and change.
    """
    two_week_change_df = compute_two_week_change(df)

    # Build list of common (uninformative) words to filter from keywords
    stopwords = get_stopwords(langs=["en"])
    common = set(stopwords) | {"skin", "product", "pretty", "love", "music", "care", "use"}

    # Rank topics by trend score
    top_items = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for topic_id, score in top_items:
        kws = get_filtered_keywords(model, topic_id, common, top_n=5)
        examples = df.loc[df['topic'] == topic_id, 'text'].dropna().head(3).tolist()
        label = generate_topic_label(kws, examples)

        # Get 2-week trend change for this topic
        change_row = two_week_change_df[two_week_change_df['TopicID'] == topic_id]
        change_value = change_row['2-week-change'].values[0] if not change_row.empty else 0.0

        rows.append({
            "TopicID": topic_id,
            "Label": label,
            "Score": round(score, 2),
            "2-week-change": round(change_value, 2)
        })

    return pd.DataFrame(rows)


def get_trends(df: pd.DataFrame, min_history: int = 2):
    """
    Full pipeline:
    - Filters for English posts.
    - Preprocesses text data.
    - Trains topic model.
    - Computes trends and bursts.
    - Labels and returns top topics.
    Returns:
        top_df: DataFrame of labeled top trends.
        topics: Topic assignments for all posts.
        trend_df: Time-distributed topic frequency data.
    """
    # Filter to only English-language posts
    df = df[df["textLanguage"] == "en"].copy()
    df = preprocess_posts(df)

    # Train topic model
    model, topics = train_topic_model(df["doc"].tolist())
    df["topic"] = topics

    # Determine number of time bins (days)
    num_days = df["date"].dt.normalize().nunique()

    # Aggregate topic frequencies over time
    trend_df = model.topics_over_time(
        docs=df["doc"].tolist(),
        topics=topics,
        timestamps=df["date"].tolist(),
        nr_bins=num_days
    )

    # Detect trends and bursts
    z_scores = detect_trends(trend_df, z_thresh=2.5, smooth_win=3, min_history=2)
    burst_scores = detect_bursts(trend_df, posts_df=df, s=2.0, gamma=1.0, min_history=2)

    # Merge trend and burst detections
    all_topics = set(z_scores) | set(burst_scores)
    combined = {
        tid: burst_scores.get(tid, z_scores.get(tid))
        for tid in all_topics
    }

    # Filter out low-volume topics
    filtered = {
        tid: score
        for tid, score in combined.items()
        if trend_df.loc[trend_df.Topic == tid, "Frequency"].sum() >= 4
    }

    # Label top trending topics
    top_df = label_top_trends(filtered, model, df)
    return top_df, topics, trend_df


def compute_two_week_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized percentage change in topic frequency between current and prior week.
    - Compares per-topic share of posts week over week.
    Returns:
        DataFrame of recent changes per topic.
    """
    df = df.copy()
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    # Count topic mentions per week
    weekly_counts = df.groupby(['week', 'topic']).size().reset_index(name='count')
    total_counts = df.groupby('week').size().reset_index(name='total')

    # Calculate weekly share of each topic
    weekly_counts = weekly_counts.merge(total_counts, on='week')
    weekly_counts['share'] = weekly_counts['count'] / weekly_counts['total']

    # Compute change relative to previous week
    weekly_counts.sort_values(['topic', 'week'], inplace=True)
    weekly_counts['prev_share'] = weekly_counts.groupby('topic')['share'].shift(1)

    weekly_counts['two_week_change'] = np.where(
        weekly_counts['prev_share'] > 0,
        (weekly_counts['share'] - weekly_counts['prev_share']) / weekly_counts['prev_share'],
        0.0
    )

    latest_week = weekly_counts['week'].max()

    # Output latest week's changes for all topics
    change_df = (
        weekly_counts[weekly_counts['week'] == latest_week][['topic', 'two_week_change']]
        .rename(columns={'topic': 'TopicID', 'two_week_change': '2-week-change'})
        .fillna(0)
    )

    return change_df