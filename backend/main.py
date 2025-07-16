import os
import pandas as pd
from backend.models.successful_posts_drivers import successful_posts_drivers
from backend.models.trends import get_trends
from backend.data_source.data_connection import load_comments_posts_transcript, load_posts_transcripts, load_hashtags_posts,load_posts_profiles, load_music
from backend.preprocessing.preprocessing import filter_by_language, detect_language, filter_by_date, filter_by_recent_days, filter_past_30_days_2_weeks_ago
from backend.models.sentiment import run_sentiment_analysis_on_comments
from backend.models.viralvideos import get_top_viral_videos
from backend.models.ingredients import analyze_ingredient_sentiments, get_top_example_comments
from backend.preprocessing.ingredientsBeiersdorf import load_ingredient_map
from backend.models.competitor_analysis import get_brand_sentiment_summary
from backend.models.viralMusic import get_top5_trending_music
from backend.models.trending_songs import trending_songs
from backend.models.topics import run_topic_model
from backend.models.hashtags import get_weekly_hashtag_trends


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_data(cache=True):
    
    #os.makedirs("data/filtered_data", exist_ok=True)
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
   
    df_music = load_music(cache=cache)
    df_music.to_csv("data/filtered_data/music.csv", index=False)


if __name__ == "__main__":
    """ 
    # Pull data
    prepare_data(cache=False)
  
    # Sentiment Analysis
    df_comments = pd.read_csv('data/filtered_data/comments_posts_transcripts.csv')
    df_comments = run_sentiment_analysis_on_comments(df=df_comments, output_path='data/filtered_data/comments_sentiment.csv', batch_size=64)
    """
    # Topic Modeling
    df_comments = pd.read_csv('data/filtered_data/comments_sentiment.csv')
    # 30-day window
    df_comments_recent_30 = filter_by_recent_days(df=df_comments, days=30)
    model_30, topic_summary_30, df_named_30 = run_topic_model(df=df_comments_recent_30, min_cluster_size=10, min_samples=3)
    topic_summary_30.to_csv("data/dashboard/topic_summary_30.csv", index=False)
    # 90-day window
    df_comments_recent_90 = filter_by_recent_days(df=df_comments, days=90)
    model_90, topic_summary_90, df_named_90 = run_topic_model(df=df_comments_recent_90)
    topic_summary_90.to_csv("data/dashboard/topic_summary_90.csv", index=False)
    """
    # Viral Videos
    top5_weekly = get_top_viral_videos("data/filtered_data/comments_posts_transcripts.csv")
    top5_weekly.to_csv("data/dashboard/top5_weekly.csv", index=False)
    # Hashtag Analysis
    hashags_result_table = get_weekly_hashtag_trends('data/filtered_data/hashtags_posts.csv')
    hashags_result_table.to_csv("data/dashboard/hashags_result_table.csv", index=False)
   
    # Ingredient Sentiment 
    
    ingredient_map = load_ingredient_map("data/static_data/Ingredient_mapping.csv") 
    ingredient_df, exploded_ingredient_df = analyze_ingredient_sentiments(
        comment_file="data/filtered_data/comments_sentiment.csv",
        ingredient_map=ingredient_map
    )
    exploded_ingredient_df.to_csv("data/dashboard/ingredients_exploded.csv", index=False)
    ingredient_df.to_csv("data/dashboard/ingredients_results.csv", index=False)


    example_comments = get_top_example_comments(exploded_ingredient_df)
    example_comments.to_csv("data/dashboard/ingredients_examplecomments.csv", index=False)

  
    # Brand Sentiment
    brands_df, brands_examples_df = get_brand_sentiment_summary("data/filtered_data/comments_posts_transcripts.csv")
    brands_df.to_csv("data/dashboard/brand_sentiment_summary.csv", index=False)
    brands_examples_df.to_csv("data/dashboard/brand_sentiment_summary_examples.csv", index=False)

    
    # Music Trends
    music = pd.read_csv("data/filtered_data/music.csv") 
    viralMusic = get_top5_trending_music(music)
    viralMusic.to_csv("data/dashboard/top5_viralMusic.csv", index=False)
  
    # Successful post drivrs
    successful_posts_drivers(input_path= "data/filtered_data/posts_transcripts.csv") 

    # Trend Detection 
    df_posts = pd.read_csv('data/filtered_data/posts_transcripts.csv')
    df_posts_recent = filter_by_recent_days(df=df_posts, days=40)
    top_df, topics, trends = get_trends(df=df_posts) # min_history=4
    top_df.to_csv("data/dashboard/trend_top.csv")
    """