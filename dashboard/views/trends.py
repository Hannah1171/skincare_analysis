# views/trends.py
import streamlit as st
import plotly.graph_objects as go
from data_loader import TIKTOK_PINK
import pandas as pd

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
    titles = ["dame Un Grrr", "morning routine", "korean skincare", "glow-up"] # ADJUST HERE 
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




