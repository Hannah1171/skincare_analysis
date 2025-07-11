#Run with the command caffeinate poetry run python -m skincare.scripts.main | tee log.txt
#will help to keep the laptop alive

#Change cache=cache back
import pandas as pd
from backend.models.trending_songs import trending_songs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from backend.models.trends import get_trends
from data_source.data_connection import load_comments_posts_transcript, load_posts_transcripts, load_hashtags_posts
from backend.preprocessing.preprocessing import filter_by_language, detect_language, filter_by_date, filter_by_recent_days
from backend.models.sentiment import run_sentiment_analysis_on_comments
from backend.models.trends_tdif import get_trending_keywords_with_tfidf
from backend.models.viralvideos import get_top_viral_videos
from backend.models.hashtags import build_weekly_normalized_table
from backend.models.ingredients import analyze_ingredient_sentiments, get_top_example_comments
from backend.preprocessing.ingredientsBeiersdorf import load_ingredient_map
from backend.models.competitor_analysis import get_brand_sentiment_summary
from backend.models.viralMusic import get_top5_trending_music

from backend.models.topics import generate_hierarchical_topics, run_topic_model
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
    df_comments = pd.read_csv('data/comments_sentiment_posts_transcripts.csv')
    df_comments_recent = filter_by_recent_days(df=df_comments, days=30)

    #df_comments_recent = run_sentiment_analysis_on_comments(df=df_comments_recent,output_path='data/comments_sentiment.csv')
    #df_comments_recent_30 = filter_by_recent_days(df=df_comments_recent, days=30)
    model, topic_summary, df_named = run_topic_model(df=df_comments_recent)
    #posts = pd.read_csv('data/posts_transcripts.csv')
    #posts = filter_by_recent_days(df=posts, days=90)

    #top_df, topics, trends = get_trends(df=posts, min_history=5)
    #top_df.to_csv("data/trend_top.csv")
    #trends.to_csv("data/trends.csv")
    #print(topics)
    #print(trends)

    #trend_tdidf = get_trending_keywords_with_tfidf(posts)
    #trend_tdidf.to_csv("data/trends_tdidf")

    topic_summary.to_csv("data/topic_summary.csv", index=False)
    #df_named.to_csv("data/df_named.csv", index=False)
    #print(df_comments_recent['text'].nunique())

    prepare_data()
    '''
    run_sentiment_analysis_on_comments(
        input_path="data/comments_posts_transcripts.csv",
        output_path="data/comments_posts_transcripts.csv",
        batch_size=64 #was 64 ritu changes it 
    ) 
    '''
    #Actually the trends

    #trends = get_trending_keywords_with_tfidf(filename="data/posts_transcripts.csv")
    #trends.to_csv("data/trends.csv", index=False)


    #top5_weekly = get_top_viral_videos("//Users/ritushetkar/env_capstone/data/comments_posts_transcripts.csv")
    #top5_weekly.to_csv("data/top5_weekly.csv", index=False)
    #top5_monthly.to_csv("data/top5_monthly.csv", index=False)

    #hashags_result_table = build_weekly_normalized_table('data/hashtags_posts.csv', min_posts=1)
    #hashags_result_table.to_csv("data/hashags_result_table.csv", index=False)

    #ingredient_map = load_ingredient_map("backend/data_source/Ingredient_Mapping.csv")

    #Process Ingredient Sentiments
    #ingredient_df, exploded_ingredient_df = analyze_ingredient_sentiments(
    #  comment_file="data/comments_posts_transcripts.csv",
    #ingredient_map=ingredient_map
    #)
    #ingredient_df.to_csv("data/ingredients_results.csv", index=False)

    #Extract Example Comments
    #example_comments = get_top_example_comments(exploded_ingredient_df)
    #example_comments.to_csv("data/ingredients_examplecomments.csv", index=False)

    brands_df, brands_examples_df = get_brand_sentiment_summary("/Users/ritushetkar/env_capstone/data/comments_posts_transcripts.csv")
    # Save the DataFrame to CSV
    brands_df.to_csv("data/brand_sentiment_summary.csv", index=False
                    )
    brands_examples_df.to_csv("data/brand_sentiment_summary_examples.csv", index=False
                    )

    music=pd.read_csv("/Users/ritushetkar/Downloads/musicViral_Combined.csv")
    viralMusic=get_top5_trending_music(music)
    viralMusic.to_csv("data/top5_viralMusic.csv", index=False
                )