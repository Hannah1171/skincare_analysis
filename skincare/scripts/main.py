from skincare.scripts.data_connection import load_comments_posts_transcript,load_posts_transcripts, load_hashtags_posts
from skincare.pipeline.preprocessing import filter_by_language, detect_language, filter_by_date
 
#Hashtags unnested
df_hashtags_posts = load_hashtags_posts(cache=False)
df_hashtags_posts_filtered = filter_by_date(df=df_hashtags_posts)
df_hashtags_posts_filtered = filter_by_language(df=df_hashtags_posts, lang_col="textLanguage")
df_hashtags_posts_filtered.to_csv("data/df_hashtags_posts.csv", index=False)

# Posts
df_posts_transcripts = load_posts_transcripts(cache=False)
df_posts_transcripts_filtered = filter_by_date(df=df_posts_transcripts)
df_posts_transcripts_filtered = filter_by_language(df=df_posts_transcripts, lang_col="textLanguage")
df_posts_transcripts_filtered.to_csv("data/posts_transcripts.csv", index=False)

# Comments unnested
df_comments_posts_transcripts = load_comments_posts_transcript(cache=False)
df_comments_posts_transcripts_filtered = filter_by_date(df=df_comments_posts_transcripts)
df_comments_posts_transcripts = detect_language(df=df_comments_posts_transcripts, text_col="comment", lang_col="comment_lang")
df_comments_posts_transcripts_filtered = filter_by_language(df=df_comments_posts_transcripts, lang_col="comment_lang")
df_comments_posts_transcripts_filtered.to_csv("data/comments_posts_transcripts.csv", index=False)

