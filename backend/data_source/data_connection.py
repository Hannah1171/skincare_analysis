from pathlib import Path
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "capstone-ai-dev"
SQL_PATH = Path(__file__).resolve().parents[2] / "backend" / "data_source" / "sql"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw_data"

client = bigquery.Client(project=PROJECT_ID)


def load_sql(filename):
    with open(SQL_PATH / filename, "r") as f:
        return f.read()

def run_query(sql_file: str) -> pd.DataFrame:
    """Run query from .sql file (no caching here)."""
    query = load_sql(sql_file)
    return client.query(query).to_dataframe()

# ---------- Raw Loaders ----------
def load_raw_comments() -> pd.DataFrame:
    return run_query("comments.sql")

def load_raw_hashtags() -> pd.DataFrame:
    return run_query("hashtags.sql")

def load_raw_posts() -> pd.DataFrame:
    return run_query("posts.sql")

def load_raw_transcripts() -> pd.DataFrame:
    return run_query("transcripts.sql")

def load_raw_profiles() -> pd.DataFrame:
    return run_query("profiles.sql")

# ---------- Merged Loaders ----------
def load_comments_posts_transcript(cache: bool = True) -> pd.DataFrame:
    output_path = CACHE_PATH / "comments_posts_transcripts_raw.csv"

    if cache and output_path.exists():
        print(f"Loaded cached merged file: {output_path}")
        return pd.read_csv(output_path)

    print("Fetching fresh data from BigQuery and merging...")

    posts = load_raw_posts()
    comments = load_raw_comments()
    transcripts = load_raw_transcripts()

    df = posts.merge(comments, on="post_id", how="left")
    df = df.merge(transcripts, on="post_id", how="left")

    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Merged file saved to: {output_path}")

    return df

def load_hashtags_posts(cache: bool = True) -> pd.DataFrame:
    output_path = CACHE_PATH / "hashtags_raw.csv"

    if cache and output_path.exists():
        print(f"Loaded cached hashtag data: {output_path}")
        return pd.read_csv(output_path)

    print("Fetching fresh hashtag data from BigQuery...")
    df = load_raw_hashtags()

    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Hashtag data saved to: {output_path}")

    return df

def load_posts_transcripts(cache: bool = True) -> pd.DataFrame:
    output_path = CACHE_PATH / "posts_transcripts_raw.csv"

    if cache and output_path.exists():
        print(f"Loaded cached merged file: {output_path}")
        return pd.read_csv(output_path)

    print("Fetching fresh data from BigQuery and merging...")
    posts = load_raw_posts()
    transcripts = load_raw_transcripts()
    df = posts.merge(transcripts, on="post_id", how="left")

    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Merged file saved to: {output_path}")

    return df

def load_posts_profiles(cache: bool = True) -> pd.DataFrame:
    output_path = CACHE_PATH / "posts_profiles.csv"

    if cache and output_path.exists():
        print(f"Loaded cached merged file: {output_path}")
        return pd.read_csv(output_path)

    print("Fetching fresh data from BigQuery and merging...")
    profiles = load_raw_profiles()

    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_path, index=False)
    print(f"Merged file saved to: {output_path}")

    return profiles

