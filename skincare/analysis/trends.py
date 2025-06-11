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

def train_topic_model(docs: list[str]):
    model = build_topic_model(min_cluster_size=5, min_samples=2, embedding_model=embedding_model)
    topics, _ = model.fit_transform(documents=docs)
    return model, topics

def detect_trends(trend_df: pd.DataFrame, z_thresh=2.5, min_history=5, min_count=5):
    scores = {}
    for tid, grp in trend_df.groupby("Topic"):
        freqs = grp["Frequency"].values
        if len(freqs) < min_history or freqs[-1] < min_count:
            continue
        mean_past = np.mean(freqs[:-1])
        std_past = np.std(freqs[:-1], ddof=0)
        z = (freqs[-1] - mean_past) / std_past if std_past != 0 else 0
        if z > z_thresh:
            scores[tid] = z
    return scores

def detect_bursts(trend_df, s=2, gamma=1.0, smooth_win=1, min_history=5):
    burst_scores = {}
    # Ensure timestamps are sorted for consistent binning
    trend_df = trend_df.sort_values("Timestamp")
    unique_ts = trend_df["Timestamp"].unique()
    
    # Precompute total posts per time bin (Timestamp)
    total_per_bin = (
        trend_df
        .groupby("Timestamp")["Frequency"]
        .sum()
        .reindex(unique_ts, fill_value=0)
        .values
    )

    for tid, grp in trend_df.groupby("Topic"):
        freqs = grp["Frequency"].values.astype(float)  # ✅ fixed here
        if len(freqs) < min_history:
            continue
        r = freqs
        d = total_per_bin[: len(r)]
        n = len(r)
        q, _, _, _ = burst_detection(r, d, n, s=s, gamma=gamma, smooth_win=smooth_win)
        if q[-1] > 0:
            burst_scores[tid] = int(q[-1])
    return burst_scores


def label_top_trends(trend_scores: dict[int, float],
                     model,
                     df: pd.DataFrame,
                     top_n: int = 10) -> pd.DataFrame:
    stopwords = get_stopwords(langs=["en"])
    common    = set(stopwords) | {"skin", "product", "care", "use"}
    top_items = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for topic_id, z_score in top_items:
        kws      = get_filtered_keywords(model, topic_id, common, top_n=5)
        examples = (df.loc[df['topic'] == topic_id, 'text']
                      .dropna()
                      .head(3)
                      .tolist())
        label    = generate_topic_label(kws, examples)
        rows.append({
            "TopicID":  topic_id,
            "Label":    label,
            "Z-score":  z_score,
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
    burst_scores = detect_bursts(trend_df, min_history=3, s=2, gamma=1.0)

    # e.g. take union, preferring burst level if available
    all_topics = set(z_scores) | set(burst_scores)
    combined   = {
        tid: burst_scores.get(tid, z_scores.get(tid))
        for tid in all_topics
    }
    top_df = label_top_trends(combined, model, df)

    return top_df, topics, trend_df
