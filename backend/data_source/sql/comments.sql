SELECT 
  id AS post_id,
  cid AS comment_id,
  createTimeISO AS comment_createTimeISO_comment,
  text AS comment,
  diggCount AS diggCount_comment,
  replyCommentTotal AS replyCommentTotal_comment,
  uniqueId AS uniqueId_comment

FROM `capstone-ai-dev.rawmart.apiy_tiktok_post_comments_id`