import pandas as pd



#----------------------------FUNCTION---------------------------------#

import pandas as pd
import re
from transformers import pipeline
from datetime import datetime
from collections import defaultdict

def analyze_ingredient_sentiments(
    comment_file: str,
    ingredient_map: dict,
    months_back: int = 3,
    min_confidence: float = 0.0,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
) -> pd.DataFrame:
    """
    Analyze ingredient mentions and sentiment in comments.

    Parameters:
        comment_file (str): Path to the CSV file containing comments.
        ingredient_map (dict): Dictionary mapping canonical ingredients to a list of keywords.
        months_back (int): How far back in time to include comments.
        min_confidence (float): Minimum sentiment confidence threshold.
        model_name (str): Hugging Face sentiment model name.

    Returns:
        pd.DataFrame: DataFrame with sentiment percentages and mention counts per ingredient.
    """
    df = pd.read_csv(comment_file)

    df = df[df["comment"].notna() & (df["playCount"] > 0)].copy()
    df["comment"] = df["comment"].astype(str).str.lower()
    df["createTimeISO"] = pd.to_datetime(df["createTimeISO"])

    cutoff_date = pd.Timestamp(datetime.today() - pd.DateOffset(months=months_back), tz="UTC")
    df = df[df["createTimeISO"] >= cutoff_date]

    df["week"] = df["createTimeISO"].dt.to_period("W").dt.start_time

    def find_ingredient(comment_text):
        matched = set()
        for canonical, terms in ingredient_map.items():
            for term in terms:
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", comment_text):
                    matched.add(canonical)
                    break
        return list(matched)

    df["matched_ingredients"] = df["comment"].apply(find_ingredient)

    sentiment_pipe = pipeline("text-classification", model=model_name, batch_size=32)

    def analyze_sentiment(text):
        try:
            result = sentiment_pipe(text, truncation=True, max_length=512)[0]
            return pd.Series({"sentiment": result["label"], "confidence": result["score"]})
        except Exception:
            return pd.Series({"sentiment": "error", "confidence": 0.0})

    df_sentiment = df[df["matched_ingredients"].str.len() > 0].copy()
    df_sentiment[["sentiment", "confidence"]] = df_sentiment["comment"].apply(analyze_sentiment)

    df_sentiment = df_sentiment[df_sentiment["confidence"] >= min_confidence]
    df_sentiment_exploded = df_sentiment.explode("matched_ingredients")

    ingredient_sentiment = (
        df_sentiment_exploded.groupby(["matched_ingredients", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    ingredient_sentiment_pivot = ingredient_sentiment.pivot(
        index="matched_ingredients", columns="sentiment", values="count"
    ).fillna(0)

    ingredient_sentiment_pivot["total_mentions"] = ingredient_sentiment_pivot.sum(axis=1)

    ingredient_sentiment_pct = ingredient_sentiment_pivot.div(
        ingredient_sentiment_pivot["total_mentions"], axis=0
    ) * 100

    ingredient_sentiment_pct["total_mentions"] = ingredient_sentiment_pivot["total_mentions"]
    ingre_results=ingredient_sentiment_pct.sort_values(by="total_mentions", ascending=False)
    ingre_results=ingre_results.reset_index()
    return ingre_results , df_sentiment_exploded



def get_top_example_comments(
    df_sentiment_exploded: pd.DataFrame,
    min_mentions: int = 20,
    valid_sentiments: list = ["positive", "neutral", "negative"]
) -> pd.DataFrame:
    """
    Select one top-confidence comment per ingredient and sentiment.

    Parameters:
        df_sentiment_exploded (pd.DataFrame): DataFrame with exploded ingredients and sentiment labels.
        min_mentions (int): Minimum number of mentions for an ingredient to be considered.
        valid_sentiments (list): List of sentiments to include.

    Returns:
        pd.DataFrame: One comment per ingredient/sentiment with highest confidence.
    """
    # Filter to valid sentiments only
    df_clean = df_sentiment_exploded[
        df_sentiment_exploded["sentiment"].isin(valid_sentiments)
    ]

    # Get ingredients with enough mentions
    ingredient_counts = df_clean["matched_ingredients"].value_counts()
    frequent_ingredients = ingredient_counts[ingredient_counts > min_mentions].index

    df_clean = df_clean[df_clean["matched_ingredients"].isin(frequent_ingredients)]

    # Select one top-confidence comment per ingredient & sentiment
    top_comments = (
        df_clean.sort_values(by="confidence", ascending=False)
        .groupby(["matched_ingredients", "sentiment"])
        .first()
        .reset_index()[["matched_ingredients", "sentiment", "comment", "confidence"]]
    )

    return top_comments.sort_values(by=["matched_ingredients", "sentiment"])
