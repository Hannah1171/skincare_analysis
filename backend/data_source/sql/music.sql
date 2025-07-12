WITH combined_posts AS (
  SELECT 
    id AS post_id, 
    createTimeISO, 
    musicMeta.musicID, 
    musicMeta.musicName, 
    musicMeta.musicAuthor, 
    musicMeta.musicOriginal, 
    musicMeta.playUrl
  FROM `capstone-ai-dev.stage.source_apify_tiktok_hashtag_posts`
  WHERE TIMESTAMP(createTimeISO) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)

  UNION ALL

  SELECT 
    id AS post_id, 
    createTimeISO, 
    musicMeta.musicID, 
    musicMeta.musicName, 
    musicMeta.musicAuthor, 
    musicMeta.musicOriginal, 
    musicMeta.playUrl 
  FROM `capstone-ai-dev.stage.source_apify_tiktok_profile_posts`
  WHERE TIMESTAMP(createTimeISO) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
)

SELECT DISTINCT *
FROM combined_posts;
