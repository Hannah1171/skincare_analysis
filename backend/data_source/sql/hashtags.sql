SELECT 
  `capstone-ai-dev.rawmart.apify_tiktok_hashtag_posts`.id as post_id,
  text,
  textLanguage,
  createTimeISO,
  isAd,
  authorMeta.nickName AS author_nickName,
  authorMeta.signature AS author_signature,
  authorMeta.fans AS author_fans,
  videoMeta.duration AS video_duration,
  webVideoUrl,
  diggCount,
  shareCount,
  playCount,
  collectCount,
  commentCount,
  isSponsored,
  hashtag.name AS hashtag_name,
  hashtag.title AS hashtag_title,
  searchHashtag.name AS searchHashtag_name,
  searchHashtag.views AS searchHashtag_views,
  locationMeta.address AS location_address

 FROM `capstone-ai-dev.rawmart.apify_tiktok_hashtag_posts` 
 LEFT JOIN UNNEST(hashtags) AS hashtag
