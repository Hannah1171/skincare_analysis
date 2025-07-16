import streamlit as st
from streamlit_option_menu import option_menu
from data_loader import load_data
from views import home, topics, trends, ingredients, successful_posts, viral_videos


def main():
    st.set_page_config(page_title="Skincare TikTok Trends", layout="wide")

    with st.sidebar:
        st.markdown("# 🧭 Navigation")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Topics", "Trends", "Ingredients & Brands", "Successful Posts", "Viral Videos"],
            icons=["house", "chat", "fire", "capsule", "rocket", "camera"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                "icon": {"color": "black", "font-size": "20px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#FFDEE9"
                },
                "nav-link-selected": {
                    "background-color": "#FE2C55",
                    "color": "white",
                    "font-weight": "bold"
                },
            }
        )

    data = load_data()
    (
        weekly, clusters_30, clusters_90, hashtags, ingredients_df,
        ingredients_example, success_general, success_author_fans,
        success_hour, success_duration, success_wordcount,
        success_isad, brands_df, brands_examples, music, top_trends
    ) = data

    if selected == "Home":
        home.display_home()
    elif selected == "Topics":
        topics.display_collapsible_topics(df_30=clusters_30, df_90=clusters_90)
    elif selected == "Trends":
        trends.display_trends(df=top_trends)
        st.markdown("""
        <br><br><br>
        """, unsafe_allow_html=True)
        trends.display_hashtag_leaderboard(df=hashtags)
        st.markdown("""
        <br><br><br>
        """, unsafe_allow_html=True)
        trends.display_top5_music(df=music)
    elif selected == "Ingredients & Brands":
        ingredients.display_ingredient_sentiment_ui(df_sentiments=ingredients_df, df_examples=ingredients_example)
        st.divider()
        ingredients.display_brand_sentiment_ui(df_sentiment=brands_df, df_examples=brands_examples)
    elif selected == "Viral Videos":
        viral_videos.show_top_videos(df=weekly, date_col="date", title="🔥 Most Viral Skincare TikToks This Week") # Please note that depending on access rights this could cause an error
    elif selected == "Successful Posts":
        successful_posts.display_successful_post_insights(
            success_general, success_author_fans, success_hour,
            success_duration, success_wordcount, success_isad
        )


if __name__ == "__main__":
    main()
