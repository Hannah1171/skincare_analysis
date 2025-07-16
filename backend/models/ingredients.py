import pandas as pd
import re
from transformers import pipeline
from datetime import datetime
from pathlib import Path

def analyze_ingredient_sentiments(
    comment_file: str,
    ingredient_map: dict,
    months_back: int = 3,
    min_confidence: float = 0.0,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze sentiment toward ingredients mentioned in TikTok comments.

    Parameters:
        comment_file (str): Path to the CSV file containing comments.
        ingredient_map (dict): Canonical ingredient -> list of keyword variants
        months_back (int): Only include comments newer than X months
        min_confidence (float): Filter out low-confidence predictions
        model_name (str): Hugging Face model to use for sentiment

    Returns:
        - summary_df: Sentiment % breakdown and mention count per ingredient
        - exploded_df: One row per (ingredient, sentiment) with confidence and comment
    """

    # Load and clean comments
    print(f"🔍 Loading comment data from: {comment_file}")
    df = pd.read_csv(comment_file)
    df = df[df["comment"].notna() & (df["playCount"] > 0)].copy()
    df["comment"] = df["comment"].astype(str).str.lower()

    # Parse comment timestamps and filter old comments
    df["createTimeISO"] = pd.to_datetime(df["createTimeISO"], errors='coerce')
    df = df[df["createTimeISO"].notna()]
    cutoff_date = pd.Timestamp(datetime.today() - pd.DateOffset(months=months_back), tz="UTC")
    df = df[df["createTimeISO"] >= cutoff_date]

    # Add week column for potential future time grouping
    df["week"] = df["createTimeISO"].dt.to_period("W").dt.start_time

    print(f"🧪 Matching ingredients in {len(df)} comments...")

    # Ingredient keyword matching using regex
    def find_ingredient(comment_text):
        matched = set()
        for canonical, terms in ingredient_map.items():
            for term in terms:
                # Match only whole-word occurrences of each term
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", comment_text):
                    matched.add(canonical)
                    break
        return list(matched)

    # Apply ingredient matching to all comments
    df["matched_ingredients"] = df["comment"].apply(find_ingredient)
    df_filtered = df[df["matched_ingredients"].str.len() > 0].copy()

    print(f"💬 Found {len(df_filtered)} comments with matched ingredients. Running sentiment...")

    # Load sentiment analysis pipeline
    sentiment_pipe = pipeline("text-classification", model=model_name, batch_size=32, truncation=True)

    # Run sentiment analysis model
    def analyze_sentiment(text):
        try:
            result = sentiment_pipe(text[:512])[0]
            return pd.Series({"sentiment": result["label"].lower(), "confidence": result["score"]})
        except Exception:
            return pd.Series({"sentiment": "error", "confidence": 0.0})

    # Classify sentiment for each matched comment
    df_filtered[["sentiment", "confidence"]] = df_filtered["comment"].apply(analyze_sentiment)

    # Remove low-confidence results
    df_filtered = df_filtered[df_filtered["confidence"] >= min_confidence]

    # Explode to one row per ingredient (since comments can mention multiple)
    df_exploded = df_filtered.explode("matched_ingredients")

    print(f"📊 Aggregating sentiment by ingredient...")

    # Aggregate sentiment counts per ingredient
    sentiment_counts = (
        df_exploded.groupby(["matched_ingredients", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to ingredient x sentiment breakdown
    pivot = sentiment_counts.pivot(index="matched_ingredients", columns="sentiment", values="count").fillna(0)
    pivot["total_mentions"] = pivot.sum(axis=1)

    # Convert counts to percentages
    sentiment_pct = pivot.div(pivot["total_mentions"], axis=0) * 100
    sentiment_pct["total_mentions"] = pivot["total_mentions"]

    # Final sorted summary
    summary = sentiment_pct.sort_values(by="total_mentions", ascending=False).reset_index()

    print("✅ Ingredient-level sentiment analysis complete.")

    return summary, df_exploded


def get_top_example_comments(
    df_sentiment_exploded: pd.DataFrame,
    min_mentions: int = 20,
    valid_sentiments: list = ["positive", "neutral", "negative"]
) -> pd.DataFrame:
    """
    For each ingredient and sentiment:
    Return the highest-confidence example comment.
    """
    # Keep only valid sentiment labels
    df_clean = df_sentiment_exploded[df_sentiment_exploded["sentiment"].isin(valid_sentiments)]

    # Filter to ingredients with at least min_mentions
    frequent_ingredients = (
        df_clean["matched_ingredients"].value_counts()
        [lambda x: x > min_mentions]
        .index
    )
    df_clean = df_clean[df_clean["matched_ingredients"].isin(frequent_ingredients)]

    # Select top-confidence comment per (ingredient, sentiment) pair
    top_comments = (
        df_clean.sort_values(by="confidence", ascending=False)
        .groupby(["matched_ingredients", "sentiment"])
        .first()
        .reset_index()[["matched_ingredients", "sentiment", "comment", "confidence"]]
    )

    return top_comments.sort_values(by=["matched_ingredients", "sentiment"])
