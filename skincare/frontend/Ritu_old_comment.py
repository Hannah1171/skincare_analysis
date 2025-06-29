
# --- Imports ---
import ast
from matplotlib import pyplot as plt, rcParams
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards 

# --- Constants ---
TIKTOK_PINK = "#FE2C55"
GREY = "#d3d3d3"
Green = "#3CB54A"
POSITIVE_COLOR = "#6FBF73"  
NEUTRAL_COLOR = "#B0B0B0"   
NEGATIVE_COLOR = "#D96C7C"  
#POSITIVE_FILL = "#6FBF737E"  
#NEGATIVE_FILL = "#D96C7C3B"  
POSITIVE_FILL = "rgba(111, 191, 115, 0.49)"  # from #6FBF737E (hex alpha 7E ≈ 49%)
NEGATIVE_FILL = "rgba(217, 108, 124, 0.23)"  # from #D96C7C3B (hex alpha 3B ≈ 23%)


rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "font.size": 12,
    "text.color": "black"
})

# --- Page Setup ---
def setup_page():
    st.set_page_config(page_title="Skincare TikTok Trends", layout="wide")

# --- Sidebar Filter ---
def sidebar_date_filter():
    st.sidebar.header("📅 Filter by Date Range")
    default_end = datetime.today()
    default_start = default_end - timedelta(days=89)
    start_date, end_date = st.sidebar.date_input(
        "Select date range",
        value=(default_start.date(), default_end.date()),
        min_value=default_start.date(),
        max_value=default_end.date()
    )
    return datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time())

# --- Data Loading ---
def load_data():
    keywords = pd.read_csv("data/keywords.csv", parse_dates=["date"])
    topics = pd.read_csv("data/dummy_topic.csv", parse_dates=["Timestamp"])
    weekly = pd.read_csv("data/top5_weekly.csv", parse_dates=["date"])
 
    clusters = pd.read_csv("data/topic_clustering.csv")
    hashtags=pd.read_csv("data/hashags_result_table.csv")
    ingredients = pd.read_csv("data/ingredients_results.csv")
    ingredients_example=pd.read_csv('data/ingredients_examplecomments.csv')

    successful_post_general = pd.read_csv("data/successful_post_range.csv") 
    successful_post_author_fans = pd.read_csv("data/shap_vs_author_fans.csv") 
    successful_post_hour_posting=pd.read_csv("data/shap_vs_hour_posting.csv") 
    successful_post_video_duration=pd.read_csv("data/shap_vs_video_duration.csv") 
    successful_post_word_count=pd.read_csv("data/shap_vs_word_count.csv") 
    successful_post_is_ad=pd.read_csv("data/shap_vs_isAd.csv") 

    brands=pd.read_csv("data/brand_sentiment_summary.csv")
    brands_examples=pd.read_csv("/Users/ritushetkar/env_capstone/data/brand_sentiment_summary_examples.csv")
    weekly["week"] = weekly["date"].dt.isocalendar().week

    music=pd.read_csv("/Users/ritushetkar/env_capstone/data/top5_viralMusic.csv")

    
    return keywords, topics, weekly, clusters, hashtags, ingredients, ingredients_example, successful_post_general, successful_post_author_fans, successful_post_hour_posting, successful_post_video_duration, successful_post_word_count, successful_post_is_ad, brands, brands_examples, music

def plot_mentions(df):
    return px.line(df, x="date", y="mentions", color="keyword",
                   color_discrete_sequence=[TIKTOK_PINK],
                   title="Mentions Over Time", template="plotly_white")

def plot_tfidf(df):
    return px.area(df, x="date", y="tfidf_score", color="keyword",
                   color_discrete_sequence=[TIKTOK_PINK],
                   title="TF-IDF Score Over Time", template="plotly_white")

def show_top_videos(df, date_col, title):
    st.header(title)
    st.badge("DISCLAIMER", color='orange')
    st.markdown(" We find the top 5 viral TikTok videos for each week based on " \
    "how many times they were viewed. It looks at when each video was posted, " \
    "organizes them by week, and then picks the five videos with the highest " \
    "play counts in that time period. This makes it easy to track which content " \
    "captured the most attention on TikTok each week, helping us understand what’s " \
    "trending and resonating with audiences over time.")
    valid = df[df["bucketUrl"].notna()].head(6)
    cols = st.columns(3)
    for i, (_, row) in enumerate(valid.iterrows()):
        col = cols[i % 3]
        with col:
            st.markdown(
                f"""
                <iframe src="{row['bucketUrl']}" width="250" height="400" style="border:none;" allow="autoplay; fullscreen"></iframe>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"**👤 {row['author_nickName']}**")
            st.markdown(f"📝 {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}")
            st.markdown(
                f"""👍 {row['diggCount']} 💬 {row['commentCount']} 🔁 {row['shareCount']}  
                ▶️ {row['playCount']} 📥 {row['collectCount']}"""
            )

def display_collapsible_topics(df: pd.DataFrame, max_quotes: int = 3):
    df = df[df["Topic"] != -1].copy()
    df["Examples"] = df["Examples"].apply(ast.literal_eval)
    total_mentions = df["Count"].sum()

    st.header("💬 What does Gen Z talk about?")
    st.badge("DISCLAIMER", color='orange')
    st.text("Please note that the topics on this page are generated automatically by AI clustering and may include noise or miss subtle nuances. This analysis covers only the posts we have collected (via specific hashtags), so some relevant content may be missing.")
    
    options = ["Last 30 days", "Last 60 days"]
    selection = st.segmented_control(
        label="Time Range",
        options=options,
    )

    cols = st.columns(2)

    for idx, (_, row) in enumerate(df.iterrows()):
        col = cols[idx % 2]
        with col:
            topic_name = row.get("NormName") or row.get("Name")
            count = int(row["Count"])
            examples = row["Examples"][:max_quotes]
            percent = min(count / total_mentions, 1.0) * 100

            with st.expander(topic_name):
                # Header section
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(
                        f"<span style='color:#ee1d52; font-weight:bold'>{count} mentions</span>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div style='background-color:#f5c0d1; border-radius:5px; height:20px; width:100%'>
                            <div style='background-color:#ee1d52; width:{percent:.2f}%; height:100%; border-radius:5px'></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Quotes section
                st.markdown("<div style='color:#ee1d52; font-size:20px; font-weight:600; margin:0px 0;'>Voice of Gen Z</div>", unsafe_allow_html=True)
                st.caption("Real comments from Tik Tok users")
                for quote in examples:
                    st.markdown(
                        f"<span style='font-size:16px; color:#ee1d52; font-weight:bold;'>&ldquo;&nbsp;</span>"
                        f"<span style='font-size:14px; color:black;'>{quote.strip()}</span>"
                        f"<span style='font-size:16px; color:#ee1d52; font-weight:bold;'>&nbsp;&rdquo;</span>",
                        unsafe_allow_html=True
                    )

                # Sentiment section
                st.markdown("<div style='color:#ee1d52; font-size:20px; font-weight:600; margin-top:6px;'>Sentiment</div>", unsafe_allow_html=True)
                st.caption("Share of comments that are positive, negative or neutral")

                labels = ["Positive", "Neutral", "Negative"]
                values = [
                    row.get("positive_share", 0) or 0,
                    row.get("neutral_share", 0) or 0,
                    row.get("negative_share", 0) or 0,
                ]
                colors = ["#6FBF73", "#B0B0B0", "#D96C7C"]
                custom_labels = [
                    f"<span style='font-weight:bold'>{label}</span><br>{int(value * 100)}%" 
                    for label, value in zip(labels, values)
                ]

                fig = go.Figure(data=[
                    go.Pie(
                        labels=custom_labels,
                        values=values,
                        marker=dict(colors=colors),
                        textinfo='label',
                        insidetextorientation='auto',
                        hoverinfo='label+percent',
                        textfont=dict(size=14, color='white')
                    )
                ])

                fig.update_layout(
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
    st.markdown("")
    st.markdown("")


def display_trends(df: pd.DataFrame):
   
    st.header("🔥 What’s Hot Next in Skincare?")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("The trends shown are based on patterns identified in the last 60 " \
    "days and aim to highlight possible shifts in content virality. Percentage changes" \
    " reflect the difference compared to the previous week. These are not definitive predictions "
    "and should not be interpreted as guarantees of future outcomes. Social media dynamics evolve " \
    "quickly and trends may shift due to platform changes, cultural developments"
    " or external events. Use these insights as directional indicators, not certainties.")
    st.markdown("")
    st.markdown("")
    titles = ["Dame Un Grrr", "Morning routine", "Korean Skincare", "Glow-up"]
    deltas = ["↑ +82%",     "↑ +28%",          "↑ +25%",           "↑ +34%"]

    cols = st.columns(4)
    for col, title, delta in zip(cols, titles, deltas):
        col.markdown(f"""
        <div style="text-align:center;">
            <span style="
            font-size:25px;
            font-weight:bold;
            color:#FE2C55;
            line-height:1.2;
            display:block;
            ">{title}</span>
            <span style="
            font-size:18px;
            color:darkgray;
            line-height:1.9;
            display:block;
            ">{delta}</span>
        </div>
        """, unsafe_allow_html=True)




def display_ingredient_sentiment_ui(df_sentiments:pd.DataFrame, df_examples:pd.DataFrame):
    
    st.header("🧪 Ingredients Gen Z talks about") #🧪💊
    st.badge("DISCLAIMER", color='orange')
    st.markdown("This function helps uncover how TikTok users feel about different " \
    "skincare ingredients. It scans through thousands of comments, finds mentions of " \
    "ingredients from a list that are interesting to Beiersdorf and then uses sentiment" \
    " analysis to label each comment as positive, neutral, or negative. It summarizes how " \
    "often each ingredient is discussed and how people feel about it and also highlights the most relevant example comments.")

    df_sentiment=df_sentiments.copy()
    df_comments=df_examples.copy()
    # Sort by total mentions
    top_ingredients = df_sentiment.sort_values("total_mentions", ascending=False).head(10)

    # Two columns layout
    col1, col2 = st.columns(2)

    for i, (_, row) in enumerate(top_ingredients.iterrows()):
        col = col1 if i % 2 == 0 else col2

        with col:
            with st.expander(f"🔍 {row['matched_ingredients']} ({int(row['total_mentions'])} mentions)"):
                percent_width = min(row["total_mentions"] / df_sentiment["total_mentions"].max(), 1.0) * 100
                st.markdown(
                    f"""
                    <div style='background-color:#f5c0d1; border-radius:6px; height:16px; width:100%; margin-bottom:10px'>
                        <div style='background-color:{TIKTOK_PINK}; width:{percent_width:.2f}%; height:100%; border-radius:6px'></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Voice of Gen Z section
                st.markdown("#### Voice of Gen Z")
                st.caption("Real comments from TikTok users")
                top_comments = df_comments[df_comments["matched_ingredients"] == row["matched_ingredients"]]
                for _, c_row in top_comments.sort_values("confidence", ascending=False).head(3).iterrows():
                    st.markdown(
                        f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&ldquo;&nbsp;</span>"
                        f"<span style='font-size:15px; color:black;'>{c_row['comment'].strip()}</span>"
                        f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&nbsp;&rdquo;</span>",
                        unsafe_allow_html=True
                    )

                # Sentiment pie chart
                st.markdown("#### Sentiment")
                st.caption("Share of comments that are positive, neutral or negative")
                sentiment_labels = ["Positive", "Neutral", "Negative"]
                sentiment_values = [
                    row.get("positive", 0),
                    row.get("neutral", 0),
                    row.get("negative", 0)
                ]
                custom_labels = [
                    f"<b>{label}</b><br>{value:.0f}%" for label, value in zip(sentiment_labels, sentiment_values)
                ]
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=custom_labels,
                            values=sentiment_values,
                            marker=dict(colors=[POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR]),
                            textinfo="label",
                            hoverinfo="label+percent",
                            textfont=dict(size=14, color="white")
                        )
                    ]
                )
                fig.update_layout(margin=dict(t=10, b=10), showlegend=False, height=280)
                st.plotly_chart(fig, use_container_width=True)





def display_successful_post_insights(successful_post_general,
                                      successful_post_author_fans,
                                      successful_post_hour_posting,
                                      successful_post_video_duration,
                                      successful_post_word_count,
                                      successful_post_is_ad):

    # 1. Horizontal bar chart of feature importance (minimalist, pink)
    st.header("🚀 What makes a post go viral?")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("Based on data of the past six months, we analyzed what drives TikTok virality. Key factors include follower count, post timing, video length, ad presence, and caption length. These elements strongly influence visibility and engagement." \
    "" \
    " However, not all success factors are captured in the data. For example, quickly replying to comments, leveraging trending sounds, and fostering community interaction also play a major role. Notably, the strongest predictor of virality is how often a post is shared, highlighting the power of social connection over pure algorithmic reach.")
    
    sorted_df = successful_post_general.sort_values("mean_abs_shap", ascending=True)
    labels = [
        "Caption Word Count",
        "Is Ad?",
        "Length of Video",
        "Time of Posting",
        "Number of Followers"
    ]

    st.subheader("Key drivers of post visibility")

    fig = go.Figure(go.Bar(
        x=sorted_df["mean_abs_shap"],
        y=sorted_df["feature"],
        orientation="h",
        marker=dict(color=TIKTOK_PINK),
        text = [f"<b>&nbsp;&nbsp;{feat}</b>" for feat in labels],
        textposition="inside",
        insidetextanchor="start",
        insidetextfont=dict(color="white", size=14),
    ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            title=dict(
                text="Feature Drivers",
                font=dict(size=18, color="black")
            ),
            showticklabels=False,
            showgrid=False
        ),
        margin=dict(t=10, b=20, l=20, r=20),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider() 

    # 2. Interactive sections per feature
    st.subheader("Optimal settings for virality")
    st.markdown("<br>", unsafe_allow_html=True)
    successful_post_general["optimal_value_range"] = successful_post_general["optimal_value_range"].apply(
    lambda x: x.replace(".0", "") if x.endswith(".0") else x
)
    cols = st.columns(5)
    labels = ["Followers", "Posting Time", "Video Length", "Caption Word Count", "Ad"]
    #ävalues = successful_post_general['optimal_value_range']
    values = ["> 500k", "18–23h", "20–30s", "16 words", "No ad"]
    
    for col, label, value in zip(cols, labels, values):
        col.metric(label=label, value=value)

    # Define label-to-feature mapping
    label_to_feature = {
        "Followers": "author_fans",
        "Posting Time": "hour_posting",
        "Video Length": "video_duration",
        "Caption Word Count": "word_count",
        "Ad": "is_ad"
    }

    feature_data_map = {
        "author_fans": successful_post_author_fans,
        "hour_posting": successful_post_hour_posting,
        "video_duration": successful_post_video_duration,
        "word_count": successful_post_word_count,
        "is_ad": successful_post_is_ad
    }

    # Selectbox with display labels
    st.markdown("""
    <div style='font-size:20px; font-weight:600; color:#000; margin-bottom:0px;'>
    Select a driver to explore
    </div>
    """, unsafe_allow_html=True)
    selected_label = st.selectbox("", list(label_to_feature.keys()))
    selected_feature = label_to_feature[selected_label]

    # Load corresponding data
    df_plot = feature_data_map[selected_feature]
    x_col, y_col = df_plot.columns

    x = df_plot[x_col].values
    y = df_plot[y_col].values

    fig = go.Figure()

    if selected_feature == "is_ad":
        x = df_plot[x_col]
        y = df_plot[y_col]
        colors = [POSITIVE_FILL if val >= 0 else NEGATIVE_FILL for val in y]

        fig = go.Figure(go.Bar(
            x=x,
            y=y,
            marker_color=colors
        ))

        fig.update_layout(
            xaxis_title="Ad (0 = No, 1 = Yes)",
            yaxis_title="Impact on Virality",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20)
        )
    else:

        # Add main line (black)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            name="Impact Line"
        ))

        # Add positive fill
        fig.add_trace(go.Scatter(
            x=x,
            y=[val if val > 0 else 0 for val in y],
            fill='tozeroy',
            mode='none',
            fillcolor=POSITIVE_FILL,
            hoverinfo='skip',
            showlegend=False
        ))

        # Add negative fill
        fig.add_trace(go.Scatter(
            x=x,
            y=[val if val < 0 else 0 for val in y],
            fill='tozeroy',
            mode='none',
            fillcolor=NEGATIVE_FILL,
            hoverinfo='skip',
            showlegend=False
        ))

        # Add horizontal reference line
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            xaxis_title=selected_label,
            yaxis_title="Impact on Virality",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False
        )

    st.plotly_chart(fig, use_container_width=True)

def display_home():
    st.image("/Users/ritushetkar/Desktop/image.png", use_container_width=True)
    st.title("✨ Welcome to the Skincare Dashboard")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("The data used for the dashboard consists of the last 9 months and was collected by scraping TikTok using the following hahstags #skincare, #skincareroutine, #hautpflege, #hautpflegeroutine.")
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Posts each month", "1.500")
    col2.metric("Comments each month", "8.500")
    col3.metric("Updated", "Weekly") 
    st.divider()
    st.markdown("### 🧑‍💻 How does GenZ talk about skincare?")
    
    st.markdown("By analyzing the most frequent word combinations (n-grams) in TikTok comments, we surfaced common phrases GenZ uses when talking about skincare. " \
    "These phrases reflect curiosity, appreciation, and a desire to connect, with both products and creators.")
    # Custom HTML table with CSS
    st.markdown("""
    <style>
    .pretty-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .pretty-table thead {
    background-color: #f5f5f5;
    color: #333;
    text-align: left;
    }
    .pretty-table th, .pretty-table td {
    padding: 12px 18px;
    border-bottom: 1px solid #eee;
    }
    .pretty-table td {
    color: #222;
    }
    </style>

    <table class="pretty-table">
    <thead>
        <tr>
        <th>Theme</th>
        <th>Insight</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td><strong>Self-expression</strong></td>
        <td>Discovery & connection</td>
        </tr>
        <tr>
        <td><strong>TikTok's Role</strong></td>
        <td>Inspiration source & beauty search engine</td>
        </tr>
        <tr>
        <td><strong>Product Queries</strong></td>
        <td>“Where to buy?” / “What’s the name?”</td>
        </tr>
        <tr>
        <td><strong>Comment Tone</strong></td>
        <td>Curious, emotional; fosters engagement</td>
        </tr>
        <tr>
        <td><strong>Affirmations</strong></td>
        <td>“love this”, “thank you” to connect with creators</td>
        </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

def display_hashtag_leaderboard(df: pd.DataFrame):
    st.subheader("Trending Hashtags Leaderboard")
    st.markdown("")
    st.markdown("")
    exclude_tags = ["skincare", "skincareroutine", "hautpflege", "hautpflegeroutine"]
    df = df[~df["hashtag_name"].str.lower().isin(exclude_tags)]
    df_sorted = df.sort_values(by="this_week_share", ascending=False).head(10).reset_index(drop=True)

    def render_row(i, row):
        trend_values = [row["week_before_last_share"], row["last_week_share"], row["this_week_share"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],
            y=trend_values,
            mode="lines+markers",
            line=dict(color="#FE2C55", width=2),
            marker=dict(size=4),
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=60,
            width=180,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        col1, col2, col3, col4, col5 = st.columns([1, 4, 2, 2, 3])
        col1.markdown(f"{i+1}")
        col2.markdown(f"{row['hashtag_name']}")
        col3.markdown(f"{row['this_week_share'] * 100:.1f}%")
        col4.markdown(f"{row['last_week_share'] * 100:.1f}%")
        col5.plotly_chart(fig, use_container_width=False)
        st.markdown("<div class='row-space'></div>", unsafe_allow_html=True)

    # Header
    col1, col2, col3, col4, col5 = st.columns([1, 4, 2, 2, 3])
    col1.markdown("**Rank**")
    col2.markdown("**Hashtags**")
    col3.markdown("**% This Week**")
    col4.markdown("**% Last Week**")
    col5.markdown("**Trend**")
    st.markdown("<hr style='margin-top: -10px; margin-bottom: 20px; border: 1px solid #ddd;' />", unsafe_allow_html=True)

    # State variable for button-based reveal
    if "show_more_hashtags" not in st.session_state:
        st.session_state.show_more_hashtags = False

    # Show first 5
    for i, row in df_sorted.head(5).iterrows():
        render_row(i, row)

    # Show button
    if not st.session_state.show_more_hashtags:
        if st.button("See more hashtags"):
            st.session_state.show_more_hashtags = True

    # Show 6–10 if button clicked
    if st.session_state.show_more_hashtags:
        for i, row in df_sorted.iloc[5:].iterrows():
            render_row(i, row)





def display_brand_sentiment_ui(df_sentiment: pd.DataFrame, df_examples: pd.DataFrame):
    st.subheader("Explore your Competitors")
    
    st.markdown("This function analyzes the posts to see how people talk about skincare brands that Beiersdorf is interested in. It checks whether the mentiosn of each brand are positive," \
    "neutral, or negative and summarizes this in a table with its share." \
    " It also picks real example posts for each brand and sentiment type, helping " \
    "you understand not just how often a brand is talked about but how people feel about it.")
   

    # --- FIX COLUMN CASE ---
    df_sentiment.columns = df_sentiment.columns.str.lower()
    df_examples.columns = df_examples.columns.str.lower()

    # --- Dropdown for brand selection ---
    available_brands = df_sentiment["matched_brand"].unique()
    selected_brand = st.selectbox("Choose a brand to explore", options=available_brands)

    row = df_sentiment[df_sentiment["matched_brand"] == selected_brand].iloc[0]

    st.markdown(f"#### Sentiment breakdown for **{selected_brand.title()}**")
    
    # Pie chart
    sentiment_labels = ["Positive", "Neutral", "Negative"]
    sentiment_values = [
        row.get("positive", 0),
        row.get("neutral", 0),
        row.get("negative", 0)
    ]
    sentiment_percentages = [
        f"<b>{label}</b><br>{int(100 * value)}%" for label, value in zip(sentiment_labels, row[["pct_positive", "pct_neutral", "pct_negative"]])
    ]
    fig = go.Figure(data=[
        go.Pie(
            labels=sentiment_percentages,
            values=sentiment_values,
            marker=dict(colors=[POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR]),
            textinfo='label',
            hoverinfo='label+percent',
            textfont=dict(size=14, color='white')
        )
    ])
    fig.update_layout(margin=dict(t=10, b=10), showlegend=False, height=280)
    st.plotly_chart(fig, use_container_width=True)

    # --- Display example comments ---
    st.markdown(f"#### Voice of Gen Z Influencers about **{selected_brand.title()}**")
    st.caption("Real Captions from TikTok Influenceer")

    top_comments = (
        df_examples[df_examples["brand"] == selected_brand]
        .drop_duplicates("text")
        .sort_values("sentiment")  # To vary sentiment types
        .groupby("sentiment")
        .head(1)
    )

    for _, comment_row in top_comments.iterrows():
        comment_text = comment_row["text"]
        transcribed = comment_row.get("transcribed_text", "")
        description = comment_row.get("video_description", "")
        video_url = comment_row.get("bucketurl", "")

        st.markdown(
            f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&ldquo;&nbsp;</span>"
            f"<span style='font-size:15px; color:black;'>{comment_text.strip()}</span>"
            f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&nbsp;&rdquo;</span>",
            unsafe_allow_html=True
        )
      

        if pd.notna(video_url) and isinstance(video_url, str) and video_url.startswith("http"):
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <iframe src="{video_url}" width="250" height="400" style="border:none;" allow="autoplay; fullscreen"></iframe>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")



def display_top5_music(df: pd.DataFrame):
    st.divider()
    st.subheader("Top 5 Trending Music This Week")
    st.markdown("")
    st.markdown("")
    df = df.head(10).reset_index(drop=True)

    if "show_more_music" not in st.session_state:
        st.session_state.show_more_music = False

    def render_music(idx, row):
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown(f"**#{idx + 1}: {row['musicName']}**")
            st.markdown(f"*by {row['musicAuthor']}*")

        with col2:
            if row['playUrl']:
                st.audio(row['playUrl'])
            else:
                st.warning("Audio not available")



    # Display top 5
    for idx, row in df.head(5).iterrows():
        render_music(idx, row)

    # Reveal next 5 on button click
    if not st.session_state.show_more_music:
        if st.button("See more music"):
            st.session_state.show_more_music = True

    if st.session_state.show_more_music:
        for idx, row in df.iloc[5:].iterrows():
            render_music(idx, row)




def main():
    st.set_page_config(page_title="Skincare TikTok Trends", layout="wide")
    

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("# 🧭 Navigation")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Topics", "Trends", "Ingredients & Brands", "Successful Posts", "Viral Videos", ],
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

    # Load data
    keywords, topics, weekly, clusters, hashtags, ingredients, ingredients_example, successful_post_general, successful_post_author_fans, successful_post_hour_posting, successful_post_video_duration, successful_post_word_count, successful_post_is_ad, brands, brands_examples, music= load_data()

    # Page Routing
    if selected == "Home":
        display_home()

    elif selected == "Topics":
        display_collapsible_topics(df=clusters)

    elif selected == "Trends":
        display_trends(df=hashtags)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        display_hashtag_leaderboard(df=hashtags)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        display_top5_music(df=music)

    elif selected == "Ingredients & Brands":
        display_ingredient_sentiment_ui(df_sentiments=ingredients, df_examples=ingredients_example)
        st.divider()
    
        display_brand_sentiment_ui(df_sentiment=brands, df_examples=brands_examples)



    elif selected == "Viral Videos":
        show_top_videos(df=weekly, date_col="date", title="🔥 Most Viral Skincare TikToks This Week")

    elif selected == "Successful Posts":
        display_successful_post_insights(successful_post_general,
                                      successful_post_author_fans,
                                      successful_post_hour_posting,
                                      successful_post_video_duration,
                                      successful_post_word_count,
                                      successful_post_is_ad)


if __name__ == "__main__":
    main()

