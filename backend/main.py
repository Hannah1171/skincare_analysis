import os
import pandas as pd
from backend.models.trends import get_trends
from backend.data_source.data_connection import load_comments_posts_transcript, load_posts_transcripts, load_hashtags_posts,load_posts_profiles
from backend.preprocessing.preprocessing import filter_by_language, detect_language, filter_by_date, filter_by_recent_days
from backend.models.sentiment import run_sentiment_analysis_on_comments
from backend.models.trends_tdif import get_trending_keywords_with_tfidf
from backend.models.viralvideos import get_top_viral_videos
from backend.models.ingredients import analyze_ingredient_sentiments, get_top_example_comments
from backend.preprocessing.ingredientsBeiersdorf import load_ingredient_map
from backend.models.competitor_analysis import get_brand_sentiment_summary
from backend.models.viralMusic import get_top5_trending_music
from backend.models.trending_songs import trending_songs
from backend.models.topics import run_topic_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_data(cache=True):
    df_hashtags = load_hashtags_posts(cache=cache)
    df_hashtags = filter_by_date(df=df_hashtags)
    df_hashtags = filter_by_language(df=df_hashtags, lang_col="textLanguage")
    df_hashtags.to_csv("data/filtered_data/hashtags_posts.csv", index=False)

    df_posts = load_posts_transcripts(cache=cache)
    df_posts = filter_by_date(df=df_posts)
    df_posts = filter_by_language(df=df_posts, lang_col="textLanguage")
    df_posts.to_csv("data/filtered_data/posts_transcripts.csv", index=False)

    df_comments = load_comments_posts_transcript(cache=cache)
    df_comments = filter_by_date(df=df_comments)
    df_comments = detect_language(df=df_comments, text_col="comment", lang_col="comment_lang")
    df_comments = filter_by_language(df=df_comments, lang_col="comment_lang")
    df_comments.to_csv("data/filtered_data/comments_posts_transcripts.csv", index=False)

    df_profiles = load_posts_profiles(cache=cache)
    df_profiles.to_csv("data/filtered_data/posts_profiles.csv", index=False)


if __name__ == "__main__":

    prepare_data()







 # --- Data Preparation ---
    prepare_data(cache=False)
    df_comments = pd.read_csv('data/comments_sentiment_posts_transcripts.csv')
    posts = pd.read_csv('data/posts_transcripts.csv')
    posts = filter_by_recent_days(df=posts, days=90)

    # --- Sentiment Analysis ---
    df_comments = run_sentiment_analysis_on_comments(df=df_comments, output_path='data/filtered_data/comments_sentiment.csv', batch_size=64)
    df_comments_recent_30 = filter_by_recent_days(df=df_comments, days=30)



    # --- Topic Modeling ---
    model, topic_summary, df_named = run_topic_model(df=df_comments_recent)
    topic_summary.to_csv("data/topic_summary.csv", index=False)
    df_named.to_csv("data/df_named.csv", index=False)
    print(df_comments_recent['text'].nunique())

    # --- Trend Detection ---
    top_df, topics, trends = get_trends(df=posts, min_history=5)
    top_df.to_csv("data/trend_top.csv")
    trends.to_csv("data/trends.csv")
    print(topics)
    print(trends)

    trend_tdidf = get_trending_keywords_with_tfidf(posts)
    trend_tdidf.to_csv("data/trends_tdidf")

    # --- Hashtag Analysis ---
    hashags_result_table = build_weekly_normalized_table('data/hashtags_posts.csv', min_posts=1)
    hashags_result_table.to_csv("data/hashags_result_table.csv", index=False)

    # --- Viral Videos ---
    top5_weekly = get_top_viral_videos("//Users/ritushetkar/env_capstone/data/comments_posts_transcripts.csv")
    top5_weekly.to_csv("data/top5_weekly.csv", index=False)
    top5_monthly.to_csv("data/top5_monthly.csv", index=False)

    # --- Ingredient Sentiment ---
    ingredient_map = load_ingredient_map("/Users/ritushetkar/Downloads/Ingredient Mapping.csv")
    ingredient_df, exploded_ingredient_df = analyze_ingredient_sentiments(
        comment_file="data/comments_posts_transcripts.csv",
        ingredient_map=ingredient_map
    )
    ingredient_df.to_csv("data/ingredients_results.csv", index=False)

    # --- Example Comments for Ingredients ---
    example_comments = get_top_example_comments(exploded_ingredient_df)
    example_comments.to_csv("data/ingredients_examplecomments.csv", index=False)

    # --- Brand Sentiment ---
    brands_df, brands_examples_df = get_brand_sentiment_summary("/Users/ritushetkar/env_capstone/data/comments_posts_transcripts.csv")
    brands_df.to_csv("data/brand_sentiment_summary.csv", index=False)
    brands_examples_df.to_csv("data/brand_sentiment_summary_examples.csv", index=False)

    # --- Music Trends ---
    music = pd.read_csv("/Users/ritushetkar/Downloads/musicViral_Combined.csv")
    viralMusic = get_top5_trending_music(music)
    viralMusic.to_csv("data/top5_viralMusic.csv", index=False)
 """