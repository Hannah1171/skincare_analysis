from skincare.scripts.data_connection import load_comments_posts_transcript,load_posts_transcripts, load_hashtags_posts
from skincare.pipeline.preprocessing import filter_by_language, detect_language, filter_by_date
from skincare.analysis.sentiment import run_sentiment_analysis_on_comments
from skincare.analysis.trends import get_trending_keywords_with_tfidf

def prepare_data(cache=False):
    # Hashtags unnested
    df_hashtags = load_hashtags_posts(cache=cache)
    df_hashtags = filter_by_date(df=df_hashtags)
    df_hashtags = filter_by_language(df=df_hashtags, lang_col="textLanguage")
    df_hashtags.to_csv("data/hashtags_posts.csv", index=False)

    # Posts
    df_posts = load_posts_transcripts(cache=cache)
    df_posts = filter_by_date(df=df_posts)
    df_posts = filter_by_language(df=df_posts, lang_col="textLanguage")
    df_posts.to_csv("data/posts_transcripts.csv", index=False)

    # Comments unnested
    df_comments = load_comments_posts_transcript(cache=cache)
    df_comments = filter_by_date(df=df_comments)
    df_comments = detect_language(df=df_comments, text_col="comment", lang_col="comment_lang")
    df_comments = filter_by_language(df=df_comments, lang_col="comment_lang")
    df_comments.to_csv("data/comments_posts_transcripts.csv", index=False)

#prepare_data()

""" run_sentiment_analysis_on_comments(
    input_path="data/comments_posts_transcripts.csv",
    output_path="data/comments_posts_transcripts.csv",
    batch_size=64
) """

keywords = get_trending_keywords_with_tfidf(filename="data/posts_transcripts.csv")
print(keywords)