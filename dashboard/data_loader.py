import pandas as pd

# Constants
TIKTOK_PINK = "#FE2C55"
GREY = "#d3d3d3"
GREEN = "#3CB54A"
POSITIVE_COLOR = "#6FBF73"
NEUTRAL_COLOR = "#B0B0B0"
NEGATIVE_COLOR = "#D96C7C"
POSITIVE_FILL = "rgba(111, 191, 115, 0.49)"
NEGATIVE_FILL = "rgba(217, 108, 124, 0.23)"


def load_data():
    weekly = pd.read_csv("data/dashboard/top5_weekly.csv", parse_dates=["date"])

    clusters = pd.read_csv("data/dashboard/topic_summary.csv")
    hashtags = pd.read_csv("data/dashboard/hashags_result_table.csv")
    ingredients = pd.read_csv("data/dashboard/ingredients_results.csv")
    ingredients_example = pd.read_csv("data/dashboard/ingredients_examplecomments.csv")

    successful_post_general = pd.read_csv("data/dashboard/successful_post_range.csv")
    successful_post_author_fans = pd.read_csv("data/dashboard/shap_vs_author_fans.csv")
    successful_post_hour_posting = pd.read_csv("data/shap_vs_hour_posting.csv") #ADJUST IF NEEDED 
    successful_post_video_duration = pd.read_csv("data/dashboard/shap_vs_video_duration.csv")
    successful_post_word_count = pd.read_csv("data/dashboard/shap_vs_word_count.csv")
    successful_post_is_ad = pd.read_csv("data/dashboard/shap_vs_isAd.csv")

    brands = pd.read_csv("data/dashboard/brand_sentiment_summary.csv")
    brands_examples = pd.read_csv("data/dashboard/brand_sentiment_summary_examples.csv")

    weekly["week"] = weekly["date"].dt.isocalendar().week

    music = pd.read_csv("data/dashboard/top5_viralMusic.csv")

    return (
        weekly, clusters, hashtags, ingredients,
        ingredients_example, successful_post_general, successful_post_author_fans,
        successful_post_hour_posting, successful_post_video_duration,
        successful_post_word_count, successful_post_is_ad,
        brands, brands_examples, music
    )
