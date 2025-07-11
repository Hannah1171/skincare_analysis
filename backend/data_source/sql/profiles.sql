SELECT
  id AS post_id,
  text,
  textLanguage,
  createTimeISO,
  isAd,
  authorMeta.nickName AS author_nickName,
  authorMeta.signature AS author_signature,
  authorMeta.fans AS author_fans,
  videoMeta[OFFSET(0)].duration AS video_duration,
  webVideoUrl,
  diggCount,
  shareCount,
  playCount,
  collectCount,
  commentCount,
  isSponsored,
  locationMeta.address AS location_address
  
FROM `capstone-ai-dev.rawmart.apify_tiktok_profile_posts`
