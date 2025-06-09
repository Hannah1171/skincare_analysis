
# --- Imports ---
import ast
from matplotlib import pyplot as plt, rcParams
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go


# --- Constants ---
TIKTOK_PINK = "#FE2C55"
GREY = "#d3d3d3"
Green = "#3CB54A"

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
    topics = pd.read_csv("/Users/ritushetkar/Downloads/dummy_topic.csv", parse_dates=["Timestamp"])
    weekly = pd.read_csv("data/top5_weekly.csv", parse_dates=["date"])
    monthly = pd.read_csv("data/top5_monthly.csv", parse_dates=["date"])
    clusters = pd.read_csv("/Users/ritushetkar/Downloads/topic_clustering.csv")

    weekly["week"] = weekly["date"].dt.isocalendar().week
    monthly["month"] = monthly["date"].dt.to_period("M")
    
    return keywords, topics, weekly, monthly, clusters

# --- Plotting Functions ---
def plot_mentions(df):
    return px.line(df, x="date", y="mentions", color="keyword",
                   color_discrete_sequence=[TIKTOK_PINK],
                   title="Mentions Over Time", template="plotly_white")

def plot_tfidf(df):
    return px.area(df, x="date", y="tfidf_score", color="keyword",
                   color_discrete_sequence=[TIKTOK_PINK],
                   title="TF-IDF Score Over Time", template="plotly_white")

# --- Display Top Videos ---
def show_top_videos(df, date_col, title):
    st.subheader(title)
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

    st.subheader("What does Gen Z talk about?")
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

def display_hashtag_leaderboard():
    hashtag_df = pd.read_csv("data/hashags_result_table.csv", parse_dates=["date"])
    hashtag_df["week"] = hashtag_df["date"].dt.isocalendar().week

    latest_week = hashtag_df["week"].max()
    latest_week_data = hashtag_df[hashtag_df["week"] == latest_week]

    last_week = latest_week - 1
    last_week_data = hashtag_df[hashtag_df["week"] == last_week][["hashtags_name", "post_count"]]

    merged = latest_week_data.merge(last_week_data, on="hashtags_name", how="left", suffixes=("", "_last"))
    merged["post_count_last"] = merged["post_count_last"].fillna(0)
    merged["change"] = merged["post_count"] - merged["post_count_last"]

    merged = merged.sort_values("post_count", ascending=False)
    merged["rank"] = merged["post_count"].rank(method="dense", ascending=False).astype(int)
    top10 = merged[merged["rank"] <= 10]

    st.sidebar.header("⚙️ Options for Hashtags")
    exclude_common = st.sidebar.checkbox("🚫 Exclude common skincare hashtags", value=True)
    excluded_tags = ['skincare', 'skincareroutine', 'hautpflege', 'hautpflegeroutine']
    if exclude_common:
        top10 = top10[~top10["hashtags_name"].isin(excluded_tags)]

    st.markdown("### 🏆 Hashtag Leaderboard")

    for _, row in top10.iterrows():
        if row["change"] > 0:
            trend_icon = "⬆"
            trend_color = "green"
        elif row["change"] < 0:
            trend_icon = "⬇"
            trend_color = "red"
        else:
            trend_icon = "➡️"
            trend_color = "gray"

        with  st.expander(f"#{row['rank']}  |  🔖 {row['hashtags_name']}  {trend_icon}"):
                st.markdown(
                    f"<span style='color:{trend_color}; font-size:22px;'>Trend: {trend_icon}</span>",
                     unsafe_allow_html=True
                    )

                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"<span style='font-size:18px'>📌 <b>{row['hashtags_name']}</b></span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<span style='font-size:18px'>{int(row['post_count'])} posts</span>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<span style='font-size:24px; color:{trend_color}'>{trend_icon}</span>", unsafe_allow_html=True)

                trend_data = hashtag_df[hashtag_df["hashtags_name"] == row["hashtags_name"]].sort_values("date")

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <h4 style="margin: 0;">Smoothed Post Count Trend: #{row['hashtags_name']}</h4>
                        <span title="This is a 3-week moving average of post count to reduce noise and reveal consistent trends.">ℹ️</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_data["date"],
                    y=trend_data["smoothed_post_count"],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=TIKTOK_PINK, width=3),
                    name='Post Count'
                ))

                fig.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
def main():
    setup_page()
    start_dt, end_dt = sidebar_date_filter()
    
    keywords, topics, weekly, monthly, clusters = load_data()
    
    # Filter data
    keywords_filtered = keywords[(keywords["date"] >= start_dt) & (keywords["date"] <= end_dt)]
    topics_filtered = topics[(topics["Timestamp"] >= start_dt) & (topics["Timestamp"] <= end_dt)]

    # Header
    st.title("🧴 Skincare TikTok Trends Dashboard")
    st.markdown(f"<h4 style='color:{TIKTOK_PINK};'>Tracking keywords, engagement, and top content</h4>", unsafe_allow_html=True)

    # Plots
    display_collapsible_topics(df=clusters)

    # NEW: Hashtag leaderboard
    display_hashtag_leaderboard()



    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_mentions(keywords_filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_tfidf(keywords_filtered), use_container_width=True)


    # Top videos
    latest_week = weekly["week"].max()
    latest_month = monthly["month"].max()
    show_top_videos(weekly[weekly["week"] == latest_week], "date", "🎬 Top Weekly Viral Videos")
    show_top_videos(monthly[monthly["month"] == latest_month], "date", "📆 Top Monthly Viral Videos")

    # Data previews
    with st.expander("🔍 View Filtered Keyword Data"):
        st.dataframe(keywords_filtered)

    with st.expander("📈 View Filtered Trend Data"):
        st.dataframe(topics_filtered)

# --- Run ---
if __name__ == "__main__":
    main()
