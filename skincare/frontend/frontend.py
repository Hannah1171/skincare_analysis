'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="Skincare TikTok Trends", layout="wide")
TIKTOK_PINK = "#FE2C55"

# --- Generate Dummy Data ---
def create_dummy_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=89)
    date_range = pd.date_range(start=start_date, end=end_date)

    keywords = ["hydrating cleanser", "niacinamide serum", "retinol cream", "sunscreen spf50", "vitamin c toner"]
    np.random.seed(42)

    data = []
    for date in date_range:
        for keyword in keywords:
            mentions = np.random.poisson(lam=5)
            tfidf_score = np.random.uniform(0.1, 0.6)
            data.append({
                "date": date,
                "keyword": keyword,
                "mentions": mentions,
                "tfidf_score": round(tfidf_score, 4)
            })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])  # Ensure datetime dtype
    df["week"] = df["date"].dt.to_period("W")
    df["week_start"] = df["week"].dt.to_timestamp()
    return df

df = create_dummy_data()

# --- Sidebar Filters ---
from datetime import datetime

# Convert to date format for Streamlit's date_input
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# Use date_input instead of slider
start_date, end_date = st.sidebar.date_input(
    label="Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Convert selected dates back to datetime for filtering
start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# Filter your dataframe
filtered_df = df[(df["date"] >= start_datetime) & (df["date"] <= end_datetime)]

selected_keyword = st.sidebar.selectbox(
    "Filter by keyword",
    options=["All"] + sorted(df["keyword"].unique())
)

# --- Dashboard Layout ---
st.title("🧴 Skincare TikTok Trends Dashboard")
st.markdown(f"<h4 style='color:{TIKTOK_PINK};'>Tracking mentions and keyword scores across 90 days</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Mentions Over Time")
    fig_mentions = px.line(
        filtered_df,
        x="date",
        y="mentions",
        color="keyword" if selected_keyword == "All" else None,
        color_discrete_sequence=[TIKTOK_PINK],
        title="Mentions Trend"
    )
    st.plotly_chart(fig_mentions, use_container_width=True)

with col2:
    st.subheader("🔠 TF-IDF Score Over Time")
    fig_tfidf = px.area(
        filtered_df,
        x="date",
        y="tfidf_score",
        color="keyword" if selected_keyword == "All" else None,
        color_discrete_sequence=[TIKTOK_PINK],
        title="TF-IDF Score Trend"
    )
    st.plotly_chart(fig_tfidf, use_container_width=True)

# --- Top Weekly Videos Section ---
st.subheader("🎬 Featured Videos (Weekly Top)")
top5_weekly = pd.read_csv("data/top5_weekly.csv")
valid_weekly = top5_weekly[top5_weekly["bucketUrl"].notna()].head(6)

cols = st.columns(3)
for i, (_, row) in enumerate(valid_weekly.iterrows()):
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
            f"""
            👍 {row['diggCount']} 💬 {row['commentCount']} 🔁 {row['shareCount']}  
            ▶️ {row['playCount']} 📥 {row['collectCount']}
            """
        )

# --- Top Monthly Videos Section ---
st.subheader("📆 Featured Videos (Monthly Top)")
top5_monthly = pd.read_csv("data/top5_monthly.csv")
valid_monthly = top5_monthly[top5_monthly["bucketUrl"].notna()].head(6)

cols = st.columns(3)
for i, (_, row) in enumerate(valid_monthly.iterrows()):
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
            f"""
            👍 {row['diggCount']} 💬 {row['commentCount']} 🔁 {row['shareCount']}  
            ▶️ {row['playCount']} 📥 {row['collectCount']}
            """
        )



# --- Raw Data View ---
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df)
'''

# --- Imports ---
import ast
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import circlify

# --- Constants ---
TIKTOK_PINK = "#FE2C55"
GREY = "#d3d3d3"
Green = "#3CB54A"
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
    monthly = pd.read_csv("data/top5_monthly.csv", parse_dates=["date"])
    clusters = pd.read_csv("data/topic_clustering.csv")

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

def display_collapsible_topics(df, max_quotes: int = 3):
    df = df[df["Topic"] != -1]
    df["Examples"] = df["Examples"].apply(ast.literal_eval)

    st.subheader("What does Gen Z talk about?")

    for _, row in df.iterrows():
        topic = row["NormName"] if pd.notna(row["NormName"]) else row["Name"]
        count = int(row["Count"])
        examples = row["Examples"][:max_quotes]

        with st.expander(f"🔹 {topic} ({count} mentions)"):
            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.markdown("**Example Comments**")
                for q in examples:
                    st.markdown(f"- *{q.strip()}*")

            with col2:
                st.markdown("**Sentiment**")
                labels = ["Positive", "Neutral", "Negative"]
                sizes = [
                    row["positive_share"] if pd.notna(row["positive_share"]) else 0,
                    row["neutral_share"] if pd.notna(row["neutral_share"]) else 0,
                    row["negative_share"] if pd.notna(row["negative_share"]) else 0
                ]
                colors = ["#28a745", "#6c757d", "#ee1d52"]

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
                ax.axis("equal")
                st.pyplot(fig)
                    
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
