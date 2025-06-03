#Run with the command caffeinate poetry run python -m skincare.scripts.main | tee log.txt
#will help to keep the laptop alive

#Change cache=cache back
from skincare.scripts.data_connection import load_comments_posts_transcript,load_posts_transcripts, load_hashtags_posts
from skincare.pipeline.preprocessing import filter_by_language, detect_language, filter_by_date
from skincare.analysis.sentiment import run_sentiment_analysis_on_comments
from skincare.analysis.trends import get_trending_keywords_with_tfidf
from skincare.analysis.viralvideos import get_top_viral_videos


def prepare_data(cache=False):
    # Hashtags unnested
    df_hashtags = load_hashtags_posts(cache=False)
    df_hashtags = filter_by_date(df=df_hashtags)
    df_hashtags = filter_by_language(df=df_hashtags, lang_col="textLanguage")
    df_hashtags.to_csv("data/hashtags_posts.csv", index=False)

    # Posts
    df_posts = load_posts_transcripts(cache=False)
    df_posts = filter_by_date(df=df_posts)
    df_posts = filter_by_language(df=df_posts, lang_col="textLanguage")
    df_posts.to_csv("data/posts_transcripts.csv", index=False)

    # Comments unnested
    df_comments = load_comments_posts_transcript(cache=False)
    df_comments = filter_by_date(df=df_comments)
    df_comments = detect_language(df=df_comments, text_col="comment", lang_col="comment_lang")
    df_comments = filter_by_language(df=df_comments, lang_col="comment_lang")
    #do sentiment
    df_comments.to_csv("data/comments_posts_transcripts.csv", index=False)

#prepare_data()

""""
run_sentiment_analysis_on_comments(
    input_path="data/comments_posts_transcripts.csv",
    output_path="data/comments_posts_transcripts.csv",
    batch_size=64
) """
#Actually the trends
keywords = get_trending_keywords_with_tfidf(filename="data/posts_transcripts.csv")
keywords.to_csv("data/keywords.csv", index=False)

top5_weekly, top5_monthly = get_top_viral_videos("data/posts_transcripts.csv")
top5_weekly.to_csv("data/top5_weekly.csv", index=False)
top5_monthly.to_csv("data/top5_monthly.csv", index=False)