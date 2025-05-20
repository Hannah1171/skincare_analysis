import pandas as pd
import re
import emoji
import stopwordsiso as stopwordsiso
from langdetect import detect
import numpy as np

# Define supported languages and stopwords
LANGS = ["en", "de"]
EXTRA_STOPWORDS = ["im", "like", "thing", "ok", "got", "youre", "omg", "pls", "tbh", "smh", "aint"]
STOPWORDS = set(
    word for lang in LANGS for word in (stopwordsiso.stopwords(lang) or [])
).union(EXTRA_STOPWORDS)

def clean_text_keep_emojis_remove_stopwords(text):
    """Lowercase, remove URLs, mentions, and stopwords, but keep emojis."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)  # remove links and mentions
    text = emoji.replace_emoji(text, replace=lambda e, data: f" {e} ")  # space-separate emojis
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9\s\U0001F600-\U0001F64F]+", " ", text)  # remove non-alphanum except emojis
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def filter_by_date(df, date_col="createTimeISO", start="2025-02-17"):
    """Filter DataFrame to include rows on or after a given start date."""
    start_dt = pd.to_datetime(start, utc=True)
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df[df[date_col] >= start_dt].copy()
    df["date"] = df[date_col].dt.date
    return df


def preprocess_text_column(df, text_col="text_comment", new_col="clean_text"):
    """Apply full text cleaning pipeline to a single column."""
    df[new_col] = df[text_col].apply(clean_text_keep_emojis_remove_stopwords)
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


#not needed?
def combine_text_columns(df, cols=("transcribed_text", "video_description"), new_col="combined_text"):
    """Combine multiple text columns into one lowercased string column."""
    df[new_col] = (
        df[cols[0]].fillna("") + " " + df[cols[1]].fillna("")
    ).str.lower()
    return df