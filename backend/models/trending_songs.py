import pandas as pd
import numpy as np
from datetime import timedelta

# 1. CONFIGURATION
RECENT_DAYS     = 7   # Rolling window for recent activity
HISTORICAL_DAYS = 60  # Rolling window for historical baseline
TREND_THRESHOLD = 0.99  # Percentile cutoff to flag top trending songs


# 2. DATA LOADING & CLEANING
def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV, parse dates, drop incomplete rows, and enforce UTC timezone.
    """
    df = pd.read_csv(path, parse_dates=["CreateTimeISO"])
    df = df.dropna(subset=["CreateTimeISO", "musicName", "musicAuthor"])
    if df.CreateTimeISO.dt.tz is None:
        df["CreateTimeISO"] = df.CreateTimeISO.dt.tz_localize("UTC")
    return df


# 3. AGGREGATION: daily play counts per song
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play counts at daily granularity per song.
    """
    df["play_date"] = df.CreateTimeISO.dt.floor("D")  # Round timestamps to day
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
    """
    Compute rolling-window statistics to quantify trend strength:
    - Recent 7-day sum of plays.
    - Historical 60-day mean of plays.
    - Ratio of recent to historical performance.
    """
    ts = (
        daily
        .pivot(index="musicName", columns="play_date", values="plays")
        .fillna(0)
        .sort_index(axis=1)
    )

    # Sum of plays over recent N days (latest column)
    recent_sum = ts.rolling(window=recent_days, axis=1).sum().iloc[:, -1]

    # Mean of plays over historical period (ignoring recent N days)
    hist_mean = (
        ts.shift(recent_days, axis=1)
          .rolling(window=hist_days, axis=1)
          .mean()
          .iloc[:, -1]
    )

    # Trend ratio: how much current activity exceeds historical average
    trend_ratio = recent_sum / (hist_mean + 1e-6)  # avoid division by zero

    feats = pd.DataFrame({
        "musicName":   recent_sum.index,
        "recent_sum":  recent_sum.values,
        "hist_mean":   hist_mean.values,
        "trend_ratio": trend_ratio.values
    })

    return feats.dropna(subset=["hist_mean"])  # Drop songs lacking history


# 5. DETECT & REPORT TRENDING SONGS
def trending_songs(path: str) -> pd.DataFrame:
    """
    Main pipeline to detect top trending songs:
    - Load raw dataset.
    - Aggregate daily plays.
    - Compute trend features.
    - Rank songs by trend ratio percentile.
    - Filter and return top trending songs.
    """
    # Load and clean dataset
    df_raw = load_and_clean(path)

    # Preserve song-to-author mapping for final reporting
    song2author = (
        df_raw[["musicName", "musicAuthor"]]
        .drop_duplicates()
        .rename(columns={"musicAuthor": "music_author"})
    )

    # Aggregate plays and build trend features
    daily = aggregate_daily(df_raw)
    feats = compute_trend_features(daily, RECENT_DAYS, HISTORICAL_DAYS)

    # Compute percentile rank of each song's trend ratio
    feats["percentile"] = feats.trend_ratio.rank(pct=True)

    # Filter to songs above configured threshold (top 1%)
    top = feats[feats.percentile >= TREND_THRESHOLD].copy()

    # Merge author information back
    top = top.merge(song2author, on="musicName", how="left")

    # Select and reorder output columns
    df_top = top.loc[:, [
        "musicName",
        "music_author",
        "recent_sum",
        "hist_mean",
        "trend_ratio",
        "percentile"
    ]].sort_values("trend_ratio", ascending=False)
    
    print(f"Detected {len(top)} trending songs (>{TREND_THRESHOLD*100:.0f}th pct).")

    # Limit to top 30 songs for reporting
    df_top = df_top.head(30)

    return df_top
