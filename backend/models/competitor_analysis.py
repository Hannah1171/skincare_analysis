import pandas as pd
import spacy
from transformers import pipeline

def create_brand_ner_model(brands):
    """
    Create a simple spaCy NER pipeline that tags brand mentions using an entity ruler.
    """
    nlp = spacy.blank("en")  # Initialize blank English model
    ruler = nlp.add_pipe("entity_ruler")  # Add entity ruler component
    patterns = [{"label": "BRAND", "pattern": brand} for brand in brands]  # One pattern per brand
    ruler.add_patterns(patterns)  # Load patterns into the entity ruler
    return nlp


def get_brand_sentiment_summary(
    input_path,
    brands=["l'oreal", "garnier", "dove", "axe", "rexona", "vaseline", "private label"],
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
):
    """
    Detect brands mentioned in social media content and analyze associated sentiment.

    Parameters:
        input_path (str): Path to CSV file with 'text', 'transcribed_text', and 'video_description' columns.
        brands (list): List of brand names to detect.
        model_name (str): Hugging Face model name for sentiment analysis.

    Returns:
        sentiment_summary (DataFrame): Sentiment counts and percentages per detected brand.
        examples_df (DataFrame): Example content per brand/sentiment.
    """

    df = pd.read_csv(input_path)

    # Combine all text fields into a unified lowercase string for detection
    df['text_combined'] = (
        df[['text', 'transcribed_text', 'video_description']]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
        .str.replace("’", "'", regex=False)
    )

    # Load simple NER model for brand detection
    nlp = create_brand_ner_model(brands)

    # Detect brands using NER
    brand_mentions = []
    for idx, row in df.iterrows():
        text = row.get('text_combined', '')
        if not isinstance(text, str):
            continue
        doc = nlp(text)
        seen = set()  # Avoid duplicate mentions per row
        for ent in doc.ents:
            if ent.label_ == 'BRAND' and ent.text not in seen:
                brand_mentions.append({
                    'index': idx,
                    'matched_brand': ent.text,
                    'text_combined': text
                })
                seen.add(ent.text)

    # If no brands found, return empty outputs
    if not brand_mentions:
        return pd.DataFrame(), pd.DataFrame()

    mentions_df = pd.DataFrame(brand_mentions)
    mentions_df = mentions_df.merge(df, left_on='index', right_index=True)

    # Rebuild combined text after merge to ensure consistency
    mentions_df['text_combined'] = (
        mentions_df[['text', 'transcribed_text', 'video_description']]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
        .str.replace("’", "'", regex=False)
    )

    # Load sentiment analysis pipeline
    sentiment_pipe = pipeline("text-classification", model=model_name, truncation=True)

    # Run sentiment analysis per detected brand context
    def get_sentiment(text):
        try:
            result = sentiment_pipe(text[:512])[0]  # Truncate input text
            return result['label'].upper(), result['score']
        except Exception:
            return "ERROR", 0.0

    mentions_df[['sentiment', 'confidence']] = mentions_df['text_combined'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # Remove rows where sentiment prediction failed
    mentions_df = mentions_df[mentions_df['sentiment'] != 'ERROR']

    # Aggregate sentiment counts per brand
    sentiment_summary = (
        mentions_df.groupby(['matched_brand', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns=str.upper)
    )

    # Ensure all sentiment columns exist
    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        if sentiment not in sentiment_summary:
            sentiment_summary[sentiment] = 0

    # Compute total mentions and sentiment shares per brand
    sentiment_summary['TOTAL'] = sentiment_summary[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
    sentiment_summary['pct_positive'] = (sentiment_summary['POSITIVE'] / sentiment_summary['TOTAL']).round(2)
    sentiment_summary['pct_neutral'] = (sentiment_summary['NEUTRAL'] / sentiment_summary['TOTAL']).round(2)
    sentiment_summary['pct_negative'] = (sentiment_summary['NEGATIVE'] / sentiment_summary['TOTAL']).round(2)

    # Collect top example comment per brand/sentiment combination
    examples = []
    for brand in sentiment_summary['MATCHED_BRAND']:
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            filtered = mentions_df[
                (mentions_df['matched_brand'] == brand) &
                (mentions_df['sentiment'] == sentiment)
            ].copy()

            if not filtered.empty:
                # Choose highest-confidence comment
                top_row = filtered.sort_values('confidence', ascending=False).iloc[0]
                examples.append({
                    'brand': brand,
                    'sentiment': sentiment,
                    'text': top_row.get('text'),
                    'transcribed_text': top_row.get('transcribed_text'),
                    'video_description': top_row.get('video_description'),
                    'bucketUrl': top_row.get('bucketUrl'),
                    'confidence': top_row.get('confidence')
                })

    examples_df = pd.DataFrame(examples)

    return sentiment_summary.sort_values(by='TOTAL', ascending=False).reset_index(drop=True), examples_df
