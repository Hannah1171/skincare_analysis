import pandas as pd
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


