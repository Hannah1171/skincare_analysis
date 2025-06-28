import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from skincare.analysis.trends import get_trends
from skincare.scripts.data_connection import load_comments_posts_transcript, load_posts_transcripts, load_hashtags_posts
from skincare.pipeline.preprocessing import filter_by_language, detect_language, filter_by_date, filter_by_recent_days
from skincare.analysis.sentiment import run_sentiment_analysis_on_comments
from skincare.analysis.trends_tdif import get_trending_keywords_with_tfidf
from skincare.analysis.topics import generate_hierarchical_topics, run_topic_model
import pandas as pd

def prepare_data(cache=True):
    df_hashtags = load_hashtags_posts(cache=cache)
    df_hashtags = filter_by_date(df=df_hashtags)
    df_hashtags = filter_by_language(df=df_hashtags, lang_col="textLanguage")
    df_hashtags.to_csv("data/hashtags_posts.csv", index=False)

    df_posts = load_posts_transcripts(cache=cache)
    df_posts = filter_by_date(df=df_posts)
    df_posts = filter_by_language(df=df_posts, lang_col="textLanguage")
    df_posts.to_csv("data/posts_transcripts.csv", index=False)

    df_comments = load_comments_posts_transcript(cache=cache)
    df_comments = filter_by_date(df=df_comments)
    df_comments = detect_language(df=df_comments, text_col="comment", lang_col="comment_lang")
    df_comments = filter_by_language(df=df_comments, lang_col="comment_lang")
    df_comments.to_csv("data/comments_posts_transcripts.csv", index=False)

if __name__ == "__main__":
    
    #prepare_data(cache=False)
    #df_comments = pd.read_csv('data/comments_sentiment_posts_transcripts.csv')
    #df_comments_recent = filter_by_recent_days(df=df_comments, days=60)
    #model, topic_summary, df_named = run_topic_model(df=df_comments_recent)

    posts = pd.read_csv('data/posts_transcripts.csv')
    posts = filter_by_recent_days(df=posts, days=90)

    top_df, topics, trends = get_trends(df=posts, min_history=5)
    top_df.to_csv("data/trend_top.csv")
    trends.to_csv("data/trends.csv")
    print(topics)
    print(trends)

    #trend_tdidf = get_trending_keywords_with_tfidf(posts)
    #trend_tdidf.to_csv("data/trends_tdidf")

    #topic_summary.to_csv("data/topic_summary.csv", index=False)
    #df_named.to_csv("data/df_named.csv", index=False)
    #print(df_comments_recent['text'].nunique())
