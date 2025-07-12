"""import pandas as pd
import re


def get_brand_sentiment_summary(output_path, brands= ['l\'oreal', 'garnier', 'dove', 'axe', 'rexona', 'vaseline', 'private label']):
    df = pd.read_csv(output_path)

    # Combine text sources into one column
    df['text_combined'] = (
        df[['text', 'transcribed_text', 'video_description']]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
        .str.replace("’", "'")
    )

    # Match brand if mentioned
    def match_brand(text):
        for brand in brands:
            if re.search(rf'\b{re.escape(brand)}\b', text):
                return brand
        return None

    df['matched_brand'] = df['text_combined'].apply(match_brand)
    df_brand_mentions = df[df['matched_brand'].notna()].copy()

    # Aggregate sentiment counts
    sentiment_summary = (
        df_brand_mentions
        .groupby(['matched_brand', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns=str.upper)
    )

    # Add totals and percentages
    if {'POSITIVE', 'NEUTRAL', 'NEGATIVE'}.issubset(sentiment_summary.columns):
        sentiment_summary['TOTAL'] = sentiment_summary[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
        sentiment_summary['pct_positive'] = (sentiment_summary['POSITIVE'] / sentiment_summary['TOTAL']).round(2)
        sentiment_summary['pct_neutral']  = (sentiment_summary['NEUTRAL']  / sentiment_summary['TOTAL']).round(2)
        sentiment_summary['pct_negative'] = (sentiment_summary['NEGATIVE'] / sentiment_summary['TOTAL']).round(2)
    else:
        sentiment_summary['TOTAL'] = 0
        sentiment_summary['pct_positive'] = 0
        sentiment_summary['pct_neutral'] = 0
        sentiment_summary['pct_negative'] = 0

    # Get example post per sentiment for each brand
     # Get example post per sentiment for each brand (ensuring different text content)
        # Step: Example post per sentiment (ensuring distinct content)
    examples = []
    for brand in sentiment_summary['MATCHED_BRAND']:
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            filtered = df_brand_mentions[
                (df_brand_mentions['matched_brand'] == brand) &
                (df_brand_mentions['sentiment'].str.upper() == sentiment)
            ].copy()

            # Create a distinct text fingerprint to ensure content variety
            filtered['text_key'] = (
                filtered['text'].fillna('') + 
                filtered['transcribed_text'].fillna('') + 
                filtered['video_description'].fillna('')
            )
            filtered = filtered.drop_duplicates(subset='text_key')

            if not filtered.empty:
                top_post = filtered.sort_values('diggCount', ascending=False).iloc[0]
                examples.append({
                    'brand': brand,
                    'sentiment': sentiment,
                    'text': top_post.get('text'),
                    'transcribed_text': top_post.get('transcribed_text'),
                    'video_description': top_post.get('video_description'),
                    'bucketUrl': top_post.get('bucketUrl')
                })

    examples_df = pd.DataFrame(examples)

    return sentiment_summary.sort_values(by='TOTAL', ascending=False).reset_index(drop=True), examples_df


"""

import pandas as pd
import spacy
from transformers import pipeline


def create_brand_ner_model(brands):
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "BRAND", "pattern": brand} for brand in brands]
    ruler.add_patterns(patterns)
    return nlp


def get_brand_sentiment_summary(
    input_path,
    brands=["l'oreal", "garnier", "dove", "axe", "rexona", "vaseline", "private label"],
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
):
    df = pd.read_csv(input_path)

    # Combine text fields
    df['text_combined'] = (
        df[['text', 'transcribed_text', 'video_description']]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
        .str.replace("’", "'", regex=False)
    )

    # Setup NER pipeline
    nlp = create_brand_ner_model(brands)

    # Detect brands using NER
    brand_mentions = []
    for idx, row in df.iterrows():
        text = row.get('text_combined', '')
        if not isinstance(text, str):
            continue
        doc = nlp(text)
        seen = set()
        for ent in doc.ents:
            if ent.label_ == 'BRAND' and ent.text not in seen:
                brand_mentions.append({
                    'index': idx,
                    'matched_brand': ent.text,
                    'text_combined': text
                })
                seen.add(ent.text)

    if not brand_mentions:
        return pd.DataFrame(), pd.DataFrame()

    mentions_df = pd.DataFrame(brand_mentions)
    mentions_df = mentions_df.merge(df, left_on='index', right_index=True)

    # Rebuild text_combined to ensure it's present after merge
    mentions_df['text_combined'] = (
    mentions_df[['text', 'transcribed_text', 'video_description']]
    .fillna('')
    .agg(' '.join, axis=1)
    .str.lower()
    .str.replace("’", "'", regex=False)
)


    # Run sentiment on matched brand context
    sentiment_pipe = pipeline("text-classification", model=model_name, truncation=True)

    def get_sentiment(text):
        try:
            result = sentiment_pipe(text[:512])[0]
            return result['label'].upper(), result['score']
        except Exception:
            return "ERROR", 0.0

    mentions_df[['sentiment', 'confidence']] = mentions_df['text_combined'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # Filter out errors
    mentions_df = mentions_df[mentions_df['sentiment'] != 'ERROR']

    # Aggregate sentiment
    sentiment_summary = (
        mentions_df.groupby(['matched_brand', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns=str.upper)
    )

    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        if sentiment not in sentiment_summary:
            sentiment_summary[sentiment] = 0

    sentiment_summary['TOTAL'] = sentiment_summary[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
    sentiment_summary['pct_positive'] = (sentiment_summary['POSITIVE'] / sentiment_summary['TOTAL']).round(2)
    sentiment_summary['pct_neutral'] = (sentiment_summary['NEUTRAL'] / sentiment_summary['TOTAL']).round(2)
    sentiment_summary['pct_negative'] = (sentiment_summary['NEGATIVE'] / sentiment_summary['TOTAL']).round(2)

    # Top example per sentiment per brand
    examples = []
    for brand in sentiment_summary['MATCHED_BRAND']:
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            filtered = mentions_df[
                (mentions_df['matched_brand'] == brand) &
                (mentions_df['sentiment'] == sentiment)
            ].copy()

            if not filtered.empty:
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
