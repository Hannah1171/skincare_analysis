import pandas as pd
from skincare.pipeline.preprocessing import get_stopwords, clean_text
from skincare.analysis.topics import (
    build_topic_model,
    get_filtered_keywords,
    generate_topic_label,
    embedding_model
)
import numpy as np
from burst_detection import burst_detection

# Monkey patch for compatibility with old code
if not hasattr(np, 'float'):
    np.float = float

def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    stopwords = get_stopwords(langs=["en"])
    df = df.copy()
    df['date'] = pd.to_datetime(df['createTimeISO'])
    df['doc'] = (
        df['text'].fillna('') + ' ' + df['transcribed_text'].fillna('')
    ).str.strip().apply(lambda t: clean_text(t, stopwords=stopwords))
    return df

def preprocess_posts2(df: pd.DataFrame) -> pd.DataFrame:
    stopwords = get_stopwords(langs=["en"])
    stopwords += ["music", "speech"]
    df = df.copy()
    df['date'] = pd.to_datetime(df['createTimeISO'])
    df['doc'] = (
        df['transcribed_text'].fillna('')
    ).str.strip().apply(lambda t: clean_text(t, stopwords=stopwords))
    return df

def train_topic_model(docs: list[str]):
    model = build_topic_model(min_cluster_size=7, min_samples=3, embedding_model=embedding_model)
    topics, _ = model.fit_transform(documents=docs)
    return model, topics

def detect_trends(trend_df: pd.DataFrame, z_thresh=2.5, min_history=5) -> dict[int, float]:
    trend_df = trend_df.copy()
    trend_df["Date"] = pd.to_datetime(trend_df["Timestamp"]).dt.normalize()

    total_per_timestamp = (
        trend_df.groupby("Date")["Frequency"]
        .sum()
        .to_dict()
    )

    scores = {}
    for tid, grp in trend_df.groupby("Topic"):
        if tid == -1:
            continue

        freqs = grp["Frequency"].values
        ts = pd.to_datetime(grp["Timestamp"]).dt.normalize()

        if len(freqs) < min_history:
            continue

        norm_freqs = np.array([
            f / total_per_timestamp.get(t, 0) if total_per_timestamp.get(t, 0) > 0 else 0
            for f, t in zip(freqs, ts)
        ])

        mean_past = np.mean(norm_freqs[:-1])
        std_past = np.std(norm_freqs[:-1], ddof=0)
        z = (norm_freqs[-1] - mean_past) / std_past if std_past != 0 else 0

        if z > z_thresh:
            scores[tid] = z

    return scores


def detect_bursts(trend_df: pd.DataFrame, s=2, gamma=1.0, min_history=5) -> dict[int, int]:
    from burst_detection import burst_detection
    import numpy as np
    import pandas as pd

    burst_scores = {}
    trend_df = trend_df.copy()
    trend_df["Date"] = pd.to_datetime(trend_df["Timestamp"]).dt.normalize()
    trend_df = trend_df.sort_values("Timestamp")

    # Total number of posts per day (not topic freq!)
    post_counts_per_date = (
        trend_df.groupby("Date")["Frequency"]
        .first()
        .groupby(level=0)
        .count()
    )

    for topic_id, group in trend_df.groupby("Topic"):
        if topic_id == -1 or len(group) < min_history:
            continue

        group = group.sort_values("Timestamp")
        frequencies = group["Frequency"].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
        ).astype(float).to_numpy()

        timestamps = pd.to_datetime(group["Timestamp"]).dt.normalize()
        total_posts = (
            post_counts_per_date
            .reindex(timestamps, fill_value=0)
            .astype(float)
            .to_numpy()
        )

        # Filter invalid entries
        valid = (total_posts > 0) & np.isfinite(frequencies) & np.isfinite(total_posts)
        r = frequencies[valid]
        d = total_posts[valid]

        if len(r) < min_history or np.all(r == 0) or np.all(d == 0):
            continue  # insufficient valid data

        try:
            q, _, _, _ = burst_detection(r, d, len(r), s=s, gamma=gamma, smooth_win=1)
            if len(q) > 0 and q[-1] > 0:
                burst_scores[topic_id] = int(q[-1])
        except Exception:
            continue  # burst_detection failed, likely due to malformed input

    return burst_scores



def label_top_trends(trend_scores: dict[int, float],
                     model,
                     df: pd.DataFrame,
                     top_n: int = 10) -> pd.DataFrame:
    stopwords = get_stopwords(langs=["en"])
    common    = set(stopwords) | {"skin", "product", "care", "use"}
    top_items = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for topic_id, score in top_items:
        kws      = get_filtered_keywords(model, topic_id, common, top_n=5)
        examples = (df.loc[df['topic'] == topic_id, 'text']
                      .dropna()
                      .head(3)
                      .tolist())
        label    = generate_topic_label(kws, examples)
        rows.append({
            "TopicID":  topic_id,
            "Label":    label,
            "Score":  score,
            "Keywords": kws
        })

    return pd.DataFrame(rows)

def get_trends(df: pd.DataFrame, min_history: int = 4):
    df = df[df["textLanguage"] == "en"].copy()
    df = preprocess_posts(df)
    model, topics = train_topic_model(df["doc"].tolist())
    df["topic"] = topics

    trend_df = model.topics_over_time(
        docs = df["doc"].tolist(),
        topics = topics,
        timestamps= df["date"].tolist(),
        nr_bins   = 10
    )

    z_scores     = detect_trends(trend_df, z_thresh=2.5, min_history=3)
    burst_scores = detect_bursts(trend_df, min_history=4, s=2, gamma=1.0)

    # e.g. take union, preferring burst level if available
    all_topics = set(z_scores) | set(burst_scores)
    combined   = {
        tid: burst_scores.get(tid, z_scores.get(tid))
        for tid in all_topics
    }
    top_df = label_top_trends(combined, model, df)

    return top_df, topics, trend_df
