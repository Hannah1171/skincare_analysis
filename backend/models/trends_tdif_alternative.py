from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from backend.preprocessing.preprocessing import preprocess_text_column, filter_by_recent_days
from collections import Counter


def extract_top_ngrams(texts, ngram_range=(2, 3), top_n=20):
    """
    Extract top-scoring n-grams using TF-IDF.
    
    Args:
        texts (list of str): Input text data.
        ngram_range (tuple): Defines the n-gram size range to extract (e.g., bigrams and trigrams).
        top_n (int): Number of top n-grams to return based on TF-IDF score.
    
    Returns:
        List of tuples (n-gram phrase, TF-IDF score).
    """
    vectorizer = TfidfVectorizer(
        max_features=100000,  # Limit total number of n-grams to manage memory.
        stop_words=None,      # No stop word filtering applied by the vectorizer itself.
        ngram_range=ngram_range  # Specify range of n-grams (default: 2-grams and 3-grams).
    )
    X = vectorizer.fit_transform(texts)  # Fit vectorizer and transform texts into TF-IDF matrix.
    scores = X.sum(axis=0).A1            # Sum TF-IDF scores across all documents for each n-gram.
    terms = vectorizer.get_feature_names_out()  # Get the actual n-gram strings.
    sorted_idx = scores.argsort()[::-1]         # Sort n-grams by descending score.
    
    return [(terms[i], round(scores[i], 4)) for i in sorted_idx[:top_n]]  # Return top n-grams with their scores.


def filter_by_occurrence(texts, top_ngrams, min_occurrence=3):
    """
    Filter n-grams to retain only those appearing at least 'min_occurrence' times in texts.
    
    Args:
        texts (list of str): Input text data.
        top_ngrams (list of tuples): Output of extract_top_ngrams function (phrase and score).
        min_occurrence (int): Minimum times the phrase must appear in the texts to be retained.
    
    Returns:
        List of tuples (phrase, TF-IDF score, frequency).
    """
    phrase_counts = Counter()
    
    # Count how many times each n-gram appears in the entire text corpus.
    for text in texts:
        for phrase, _ in top_ngrams:
            if phrase in text:
                phrase_counts[phrase] += 1

    # Return only n-grams meeting the minimum frequency requirement.
    return [
        (phrase, score, phrase_counts[phrase])
        for phrase, score in top_ngrams
        if phrase_counts[phrase] >= min_occurrence
    ]


def get_trending_keywords_with_tfidf(df, number_of_days=28):
    """
    Main function to compute trending keywords from transcribed text using TF-IDF,
    filtered by n-gram frequency, grouped by week.
    
    Args:
        df (pd.DataFrame): Dataframe containing at least 'createTimeISO' and 'transcribed_text' columns.
        number_of_days (int): Time window (in days) to consider for analysis.
    
    Returns:
        pd.DataFrame: Keywords, TF-IDF scores, and mention counts aggregated by week.
    """
    # Filter dataset to only recent posts within the specified timeframe.
    recent_posts = filter_by_recent_days(df=df, days=number_of_days)

    # Create additional columns to group posts by week.
    recent_posts["week_date"] = pd.to_datetime(recent_posts["createTimeISO"]).dt.to_period("W").dt.to_timestamp()
    recent_posts["week_number"] = pd.to_datetime(recent_posts["createTimeISO"]).dt.to_period("W")

    all_weekly_results = []

    # Process each week separately.
    for week, group in recent_posts.groupby("week_number"):
        group["transcribed_text"] = group["transcribed_text"].astype(str)

        # Preprocess the transcribed text (cleaning, stopwords removal, etc.).
        df_filtered = preprocess_text_column(df=group, text_col='transcribed_text', new_col='transcribed_text')
        texts = df_filtered["transcribed_text"].dropna().astype(str).tolist()

        # Step 1: Extract top-scoring n-grams via TF-IDF.
        top_ngrams = extract_top_ngrams(texts, ngram_range=(2, 3), top_n=50)

        # Step 2: Filter out n-grams occurring fewer than 5 times in total.
        filtered_top_ngrams = filter_by_occurrence(texts, top_ngrams=top_ngrams, min_occurrence=5)

        # Step 3: Collect results for current week.
        for keyword, tfidf_score, mentions in filtered_top_ngrams:
            all_weekly_results.append({
                "date": week,
                "keyword": keyword,
                "tfidf_score": tfidf_score,
                "mentions": mentions
            })

    # Compile all weekly keyword statistics into a DataFrame.
    return pd.DataFrame(all_weekly_results)
