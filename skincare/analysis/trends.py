from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from skincare.pipeline.preprocessing import preprocess_text_column, filter_by_recent_days
from collections import Counter


def extract_top_ngrams(texts, ngram_range=(2, 3), top_n=20):
    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words=None,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    sorted_idx = scores.argsort()[::-1]
    return [(terms[i], round(scores[i], 4)) for i in sorted_idx[:top_n]]


def filter_by_occurrence(texts, top_ngrams, min_occurrence=3):
    phrase_counts = Counter()
    for text in texts:
        for phrase, _ in top_ngrams:
            if phrase in text:
                phrase_counts[phrase] += 1

    # Now return (phrase, tfidf_score, frequency)
    return [
        (phrase, score, phrase_counts[phrase])
        for phrase, score in top_ngrams
        if phrase_counts[phrase] >= min_occurrence
    ]


def get_trending_keywords_with_tfidf(filename, number_of_days=28):
    df = pd.read_csv(f"{filename}")
    recent_posts = filter_by_recent_days(df=df, days=number_of_days)

    # Add week column
    recent_posts["week_date"] = pd.to_datetime(recent_posts["createTimeISO"]).dt.to_period("W").dt.to_timestamp()
    recent_posts["week_number"] = pd.to_datetime(recent_posts["createTimeISO"]).dt.to_period("W")

    all_weekly_results = []

    for week, group in recent_posts.groupby("week"):
        df_filtered = preprocess_text_column(df=group, text_col='transcribed_text', new_col='transcribed_text')
        texts = df_filtered["transcribed_text"].dropna().astype(str).tolist()

        top_ngrams = extract_top_ngrams(texts, ngram_range=(2, 3), top_n=50)
        filtered_top_ngrams = filter_by_occurrence(texts, top_ngrams=top_ngrams, min_occurrence=5)

        for keyword, tfidf_score, mentions in filtered_top_ngrams:
            all_weekly_results.append({
                "date": week,
                "keyword": keyword,
                "tfidf_score": tfidf_score,
                "mentions": mentions
            })

    return pd.DataFrame(all_weekly_results)
