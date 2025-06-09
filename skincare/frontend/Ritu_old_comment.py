
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Constants ---
TIKTOK_PINK = "#FE2C55"

# --- Page Setup ---
st.set_page_config(page_title="Hashtag Leaderboard", layout="wide")
st.title("📈 Weekly Trending Hashtags")

# --- Load Data ---
hashtag_df = pd.read_csv("data/hashags_result_table.csv", parse_dates=["date"])
hashtag_df["week"] = hashtag_df["date"].dt.isocalendar().week

# --- Filter by Latest Week ---
latest_week = hashtag_df["week"].max()
latest_week_data = hashtag_df[hashtag_df["week"] == latest_week]

# --- Get Last Week for Trend Comparison ---
last_week = latest_week - 1
last_week_data = hashtag_df[hashtag_df["week"] == last_week][["hashtags_name", "post_count"]]

# --- Merge and Compute Change ---
merged = latest_week_data.merge(last_week_data, on="hashtags_name", how="left", suffixes=("", "_last"))
merged["post_count_last"] = merged["post_count_last"].fillna(0)
merged["change"] = merged["post_count"] - merged["post_count_last"]

# --- Rank Hashtags with Dense Method ---
merged = merged.sort_values("post_count", ascending=False)
merged["rank"] = merged["post_count"].rank(method="dense", ascending=False).astype(int)

# --- Filter Top 10 Ranks ---
top10 = merged[merged["rank"] <= 10]

# --- Sidebar Filter ---
st.sidebar.header("⚙️ Options")
exclude_common = st.sidebar.checkbox("🚫 Exclude common skincare hashtags", value=True)
excluded_tags = ['skincare', 'skincareroutine', 'hautpflege', 'hautpflegeroutine']
if exclude_common:
    top10 = top10[~top10["hashtags_name"].isin(excluded_tags)]

# --- Display Table with Expandable Trend ---
# --- Display Table with Expandable Trend ---
st.markdown("### 🏆 Hashtag Leaderboard")

for _, row in top10.iterrows():
    # Determine trend icon
    if row["change"] > 0:
        trend_icon = "⬆"
        trend_color = "green"
    elif row["change"] < 0:
        trend_icon = "⬇"
        trend_color = "red"
    else:
        trend_icon = "➡️"
        trend_color = "gray"

    with st.expander(f"#{row['rank']}  |  🔖 {row['hashtags_name']} {trend_icon}"):

        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"<span style='font-size:18px'>📌 <b>{row['hashtags_name']}</b></span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span style='font-size:18px'>{int(row['post_count'])} posts</span>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<span style='font-size:24px; color:{trend_color}'>{trend_icon}</span>", unsafe_allow_html=True)

        # Trend chart
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
            title=f"Smoothed Post Count Trend: #{row['hashtags_name']}",
            template="plotly_white",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
