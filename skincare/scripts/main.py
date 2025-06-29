#Run with the command caffeinate poetry run python -m skincare.scripts.main | tee log.txt
#will help to keep the laptop alive

#Change cache=cache back
from skincare.analysis.trending_songs import trending_songs
from skincare.scripts.data_connection import load_comments_posts_transcript,load_posts_transcripts, load_hashtags_posts
from skincare.pipeline.preprocessing import filter_by_language, detect_language, filter_by_date
#from skincare.analysis.sentiment import run_sentiment_analysis_on_comments
#from skincare.analysis.trends import get_trending_keywords_with_tfidf
#from skincare.analysis.viralvideos import get_top_viral_videos
#from skincare.analysis.hashtags import build_weekly_normalized_table
#from skincare.analysis.ingredients import analyze_ingredient_sentiments, get_top_example_comments
#from skincare.pipeline.ingredientsBeiersdorf import load_ingredient_map

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
''''
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

#ingredient_map = load_ingredient_map("/Users/ritushetkar/Downloads/Ingredient Mapping.csv")

# Process Ingredient Sentiments
#ingredient_df, exploded_ingredient_df = analyze_ingredient_sentiments(
 #   comment_file="data/comments_posts_transcripts.csv",
  #  ingredient_map=ingredient_map
#)
#ingredient_df.to_csv("data/ingredients_results.csv", index=False)

# Extract Example Comments
#example_comments = get_top_example_comments(exploded_ingredient_df)
#example_comments.to_csv("data/ingredients_examplecomments.csv", index=False)
song_df = trending_songs(path="data/song_data_final.csv")
song_df.to_csv("data/trending_songs.csv")