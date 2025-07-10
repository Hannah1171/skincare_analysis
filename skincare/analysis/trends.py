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
from scipy.stats import median_abs_deviation


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
    model = build_topic_model(min_cluster_size=11, min_samples=6, embedding_model=embedding_model, ngram_range=(1,2))
    topics, _ = model.fit_transform(documents=docs)
    return model, topics


def detect_trends(
    trend_df: pd.DataFrame,
    z_thresh: float  = 3.5,
    smooth_win: int    = 3,
    min_history: int   = 4
) -> dict[int, float]:
    df = trend_df.copy()
    df["day"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    total_per_day = df.groupby("day")["Frequency"].sum().to_dict()

    scores = {}
    for topic_id, grp in df.groupby("Topic"):
        if topic_id == -1 or len(grp) < min_history:
            continue

        grp    = grp.sort_values("day")
        dates  = grp["day"].to_numpy()
        freqs  = grp["Frequency"].astype(float).to_numpy()

        # 1) mask out any days where total_per_day is zero
        valid_mask = [total_per_day.get(d, 0) > 0 for d in dates]
        if sum(valid_mask) < min_history:
            continue

        dates = dates[valid_mask]
        freqs = freqs[valid_mask]

        # 2) now compute ratios only on valid days
        ratios = np.array([freq / total_per_day[d] for freq, d in zip(freqs, dates)])

        # 3) smooth
        series   = pd.Series(ratios, index=dates)
        smoothed = series.rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()

        # 4) robust z‐score
        hist = smoothed[:-1]
        med  = np.median(hist)
        mad  = median_abs_deviation(hist, scale="normal") or 1
        z    = (smoothed[-1] - med) / mad

        if z > z_thresh:
            scores[topic_id] = float(z)

    return scores



def detect_bursts(
    trend_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    s: float           = 2.0,
    gamma: float       = 1.0,
    min_history: int   = 4
) -> dict[int, int]:
    # 1) true total posts per day
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
            continue

        grp = grp.sort_values("day")
        # raw counts for the topic in each bin
        freqs = grp["Frequency"] \
                    .apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x) \
                    .astype(float) \
                    .to_numpy()

        # align to true daily totals
        days = grp["day"]
        ds  = daily_posts.reindex(days, fill_value=0).to_numpy()

        # filter out invalid days
        valid = (ds > 0) & np.isfinite(freqs)
        r, d  = freqs[valid], ds[valid]
        if len(r) < min_history or r.sum() == 0 or d.sum() == 0:
            continue

        # 4) Kleinberg burst detection
        try:
            q, _, _, _ = burst_detection(r, d, len(r), s=s, gamma=gamma, smooth_win=1)
            if q and q[-1] > 0:
                burst_scores[topic_id] = int(q[-1])
        except Exception:
            continue

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

    num_days = df["date"].dt.normalize().nunique()


    trend_df = model.topics_over_time(
        docs = df["doc"].tolist(),
        topics = topics,
        timestamps= df["date"].tolist(),
        nr_bins   = num_days
    )

    z_scores     = detect_trends( trend_df,
                              z_thresh=3.5,
                              smooth_win=3,
                              min_history=3 )
    burst_scores = detect_bursts(trend_df,
                            posts_df=df,
                            s=2.0,
                            gamma=1.0,
                            min_history=4)

    # e.g. take union, preferring burst level if available
    all_topics = set(z_scores) | set(burst_scores)
    combined   = {
        tid: burst_scores.get(tid, z_scores.get(tid))
        for tid in all_topics
    }

    # filter out any topic whose total raw count < MIN_VOLUME
    filtered = {
        tid: score
        for tid, score in combined.items()
        if trend_df.loc[trend_df.Topic == tid, "Frequency"].sum() >= 10 #min volume
    }

    top_df = label_top_trends(filtered, model, df)

    return top_df, topics, trend_df
