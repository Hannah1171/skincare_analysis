import pandas as pd
import numpy as np
from datetime import timedelta

# 1. CONFIGURATION
RECENT_DAYS     = 7 #7
HISTORICAL_DAYS = 60 #60
TREND_THRESHOLD = 0.99


# 2. DATA LOADING & CLEANING
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["CreateTimeISO"])
    df = df.dropna(subset=["CreateTimeISO", "musicName", "musicAuthor"])
    if df.CreateTimeISO.dt.tz is None:
        df["CreateTimeISO"] = df.CreateTimeISO.dt.tz_localize("UTC")
    return df


# 3. AGGREGATION: daily play counts per song
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["play_date"] = df.CreateTimeISO.dt.floor("D")
    return (
        df
        .groupby(["musicName", "play_date"])
        .size().rename("plays")
        .reset_index()
    )


# 4. FEATURE ENGINEERING: rolling-window statistics
def compute_trend_features(
    daily: pd.DataFrame,
    recent_days: int,
    hist_days: int
) -> pd.DataFrame:
    ts = (
        daily
        .pivot(index="musicName", columns="play_date", values="plays")
        .fillna(0)
        .sort_index(axis=1)
    )
    recent_sum  = ts.rolling(window=recent_days, axis=1).sum().iloc[:, -1]
    hist_mean   = ts.shift(recent_days, axis=1)\
                    .rolling(window=hist_days, axis=1).mean().iloc[:, -1]
    trend_ratio = recent_sum / (hist_mean + 1e-6)

    feats = pd.DataFrame({
        "musicName":   recent_sum.index,
        "recent_sum":  recent_sum.values,
        "hist_mean":   hist_mean.values,
        "trend_ratio": trend_ratio.values
    })
    return feats.dropna(subset=["hist_mean"])


# 5. RANK & THRESHOLD + MERGE AUTHOR
def trending_songs(path: str) -> pd.DataFrame:
    # load & clean
    df_raw = load_and_clean(path)

    # keep song→author mapping
    song2author = (
        df_raw[["musicName", "musicAuthor"]]
        .drop_duplicates()
        .rename(columns={"musicAuthor": "music_author"})
    )

    # build trend features
    daily = aggregate_daily(df_raw)
    feats = compute_trend_features(daily, RECENT_DAYS, HISTORICAL_DAYS)

    # rank into percentile
    feats["percentile"] = feats.trend_ratio.rank(pct=True)

    # filter top
    top = feats[feats.percentile >= TREND_THRESHOLD].copy()

    # merge author
    top = top.merge(song2author, on="musicName", how="left")

    # select + reorder columns
    df_top = top.loc[:, [
        "musicName",
        "music_author",
        "recent_sum",
        "hist_mean",
        "trend_ratio",
        "percentile"
    ]].sort_values("trend_ratio", ascending=False)
    
    print(f"Detected {len(top)} trending songs (>{TREND_THRESHOLD*100:.0f}th pct).")
    df_top = df_top.head(30)

    return df_top


