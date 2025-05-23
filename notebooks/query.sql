

WITH hashtag_posts as (
SELECT
  id,
  text,
  textLanguage,
  createTimeISO,
  isAd,
  isMuted,
  diggCount,
  shareCount,
  playCount,
  collectCount,
  commentCount,
  isSlideshow,
  isPinned,
  isSponsored,
  authorMeta_nickName,
  authorMeta_verified,
  authorMeta_signature,
  authorMeta_privateAccount,
  authorMeta_ttSeller,
  authorMeta_fans,
  authorMeta_heart,
  authorMeta_video,
  authorMeta_digg,
  authorMeta_region,
  authorMeta_musicName,
  authorMeta_musicAuthor,
  videoMeta_duration,
  videoMeta_definition,
  detailedMentions_nickName,
  hashtags_name,
  effectStickers_name,
  effectStickers_stickerStats_useCount,
  searchHashtag_views,
  searchHashtag_name,
  locationMeta_address
FROM `rawmart.apify_tiktok_hashtag_posts_unnested`

UNION ALL

SELECT
  id,
  text,
  textLanguage,
  createTimeISO,
  isAd,
  NULL AS isMuted,
  diggCount,
  shareCount,
  playCount,
  collectCount,
  commentCount,
  isSlideshow,
  isPinned,
  isSponsored,
  authorMeta_nickName,
  authorMeta_verified,
  authorMeta_signature,
  authorMeta_privateAccount,
  authorMeta_ttSeller,
  authorMeta_fans,
  authorMeta_heart,
  authorMeta_video,
  authorMeta_digg,
  authorMeta_region,
  musicMeta_musicName as authorMeta_musicName,
  musicMeta_musicAuthor as authorMeta_musicAuthor,
  videoMeta_duration,
  videoMeta_definition,
  detailedMentions_nickName,
  hashtags_name,
  effectStickers_name,
  effectStickers_stickerStats_useCount,
  NULL as searchHashtag_views,
  NULL as searchHashtag_name,
  locationMeta_address
FROM `rawmart.apify_tiktok_profile_posts_unnested`
)

SELECT * 
FROM hashtag_posts
LEFT JOIN `capstone-ai-dev.rawmart.apiy_tiktok_post_comments_id` AS post_comments
ON hashtag_posts.id = post_comments.id;

