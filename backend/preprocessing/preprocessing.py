import pandas as pd
import re
import emoji
import stopwordsiso as stopwordsiso
from langdetect import detect
import numpy as np

# Define supported languages and stopwords
LANGS = ["en", "de"]

def get_stopwords(langs=None, extra=None):
    """
    Build a multilingual stopword set with optional extra filler terms.
    
    Returns:
        set: A unified set of stopwords.
    """
    if langs is None:
        langs = LANGS
    if extra is None:
        extra = ["im", "like", "thing", "ok", "got", "youre", "omg", "pls", "tbh", "smh", "aint"]

    stopword_set = set()
    for lang in langs:
        words = stopwordsiso.stopwords(lang)
        if words:
            stopword_set.update(words)

    stopword_set.update(extra)
    return list(stopword_set)


def clean_text(text, stopwords=None):
    """
    Lowercase, remove URLs/mentions, normalize repeated characters,
    keep emojis, clean punctuation, and optionally remove stopwords.
    
    Args:
        text (str): Raw input text.
        stopwords (set, optional): If provided, stopwords will be removed.

    Returns:
        str: Cleaned text.
    """
    # Lowercase
    text = text.lower()

    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)

    # Keep emojis by spacing them
    text = emoji.replace_emoji(text, replace=lambda e, _: f" {e} ")

    # Remove all characters except letters, numbers, whitespace, and emojis
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9\s\U0001F600-\U0001F64F]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and normalize repeated characters
    tokens = text.split()
    tokens = [re.sub(r"(.)\1{2,}", r"\1", t) for t in tokens]

    # Remove stopwords and short tokens
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    else:
        tokens = [t for t in tokens if len(t) > 2]

    return " ".join(tokens)


def filter_by_date(df, date_col="createTimeISO", start="2025-02-17"):
    """Filter DataFrame to include rows on or after a given start date."""
    start_dt = pd.to_datetime(start, utc=True)
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df[df[date_col] >= start_dt].copy()
    df["date"] = df[date_col].dt.date
    return df

def filter_by_recent_days(df, date_col="createTimeISO", days=30):
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    df = df[df[date_col] >= cutoff].copy()
    df["date"] = df[date_col].dt.date
    return df

def preprocess_text_column(df, text_col="text_comment", new_col="clean_text"):
    """Apply full text cleaning pipeline to a single column."""
    df[new_col] = df[text_col].apply(clean_text)
    return df

def detect_language(df, text_col="text_comment", lang_col="lang"):
    """Adds a language detection column to the DataFrame."""
    def safe_detect(text):
        try:
            return detect(text)
        except:
            return np.nan

    df[lang_col] = df[text_col].apply(safe_detect)
    return df

def filter_by_language(df, lang_col="lang", allowed=("en", "de")):
    """Filters DataFrame to rows where language column is in allowed list."""
    return df[df[lang_col].isin(allowed)].copy()

def filter_short_comments(comments, min_words=4):
    """Remove comments that are too short."""
    return [c for c in comments if len(c.split()) >= min_words]

def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    hashtags = re.findall(r'#\w+', text.lower())
    return [tag.replace('#', '') for tag in hashtags]


#not needed?
def combine_text_columns(df, cols=("transcribed_text", "video_description"), new_col="combined_text"):
    """Combine multiple text columns into one lowercased string column."""
    df[new_col] = (
        df[cols[0]].fillna("") + " " + df[cols[1]].fillna("")
    ).str.lower()
    return df