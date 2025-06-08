import pandas as pd
from transformers import pipeline

def run_sentiment_analysis_on_comments(
    input_path: str,
    output_path: str,
    text_col: str = "comment",
    sentiment_col: str = "sentiment",
    confidence_col: str = "sentiment_confidence",
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Loads comments from a CSV, applies sentiment analysis, and saves results to a new CSV.

    Parameters:
        input_path (str): Path to the input CSV.
        output_path (str): Path to save the updated DataFrame.
        text_col (str): Name of the column containing text to analyze.
        sentiment_col (str): Output column for sentiment labels.
        confidence_col (str): Output column for sentiment confidence scores.
        model_name (str): HuggingFace model name for sentiment analysis.
        batch_size (int): Batch size for pipeline processing.

    Returns:
        pd.DataFrame: The full DataFrame with sentiment columns added.
    """
    df = pd.read_csv(input_path)
    texts = df[text_col].fillna("").astype(str).tolist()

    pipe = pipeline("text-classification", model=model_name, batch_size=batch_size)
    results = pipe(texts, truncation=True, batch_size=batch_size)

    sentiment_df = pd.DataFrame([
        {sentiment_col: r["label"], confidence_col: r["score"]}
        for r in results
    ])

    df = df.reset_index(drop=True)
    df[[sentiment_col, confidence_col]] = sentiment_df
    df.to_csv(output_path, index=False)

    print(f"Saved sentiment results to {output_path}")
    return df
