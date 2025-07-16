import streamlit as st
import ast
import plotly.graph_objects as go
from data_loader import TIKTOK_PINK, POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR
import pandas as pd

def display_collapsible_topics(df_30: pd.DataFrame, df_90: pd.DataFrame, max_quotes: int = 3):
    """
    Display AI-generated discussion topics from TikTok data, filtered by last 30 or 90 days.
    Each topic is shown as a collapsible block with:
    - Total mentions,
    - Example TikTok user comments,
    - Sentiment breakdown (positive, neutral, negative).
    """

    # Display main title and disclaimer for context
    st.header("💬 What does Gen Z talk about?")
    st.badge("DISCLAIMER", color='orange')
    st.text(
        "Please note that the topics on this page are generated automatically by AI clustering and may include noise "
        "or miss subtle nuances. This analysis covers only the posts we have collected (via specific hashtags), so "
        "some relevant content may be missing."
    )

    # Radio button to select dataset (30 days or 90 days)
    options = ["Last 30 days", "Last 90 days"]
    selection = st.radio(
        label="Select Time Range:",
        options=options,
        index=1  # Show "Last 90 days" by default
    )

    # Select dataframe according to user choice
    if selection == "Last 30 days":
        df = df_30.copy()
    else:
        df = df_90.copy()

    # Filter out noise topics (Topic == -1)
    df = df[df["Topic"] != -1].copy()

    # Convert example comments from string to list
    df["Examples"] = df["Examples"].apply(ast.literal_eval)

    # Calculate total mentions for percentage calculations
    total_mentions = df["Count"].sum()

    # Prepare a 2-column layout for displaying topics
    cols = st.columns(2)

    # Iterate over each topic and display it in alternating columns
    for idx, (_, row) in enumerate(df.iterrows()):
        col = cols[idx % 2]
        with col:
            topic_name = row.get("Name") or row.get("Name")
            count = int(row["Count"])
            examples = row["Examples"][:max_quotes]  # Limit number of displayed example quotes
            percent = min(count / total_mentions, 1.0) * 100  # Calculate share percentage

            # Create collapsible section per topic
            with st.expander(topic_name):

                # Display total mentions and progress bar
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

                # Display example TikTok user quotes
                st.markdown(
                    "<div style='color:#ee1d52; font-size:20px; font-weight:600; margin:0px 0;'>Voice of Gen Z</div>",
                    unsafe_allow_html=True
                )
                st.caption("Real comments from TikTok users")
                for quote in examples:
                    st.markdown(
                        f"<span style='font-size:16px; color:#ee1d52; font-weight:bold;'>&ldquo;&nbsp;</span>"
                        f"<span style='font-size:14px; color:black;'>{quote.strip()}</span>"
                        f"<span style='font-size:16px; color:#ee1d52; font-weight:bold;'>&nbsp;&rdquo;</span>",
                        unsafe_allow_html=True
                    )

                # Sentiment breakdown section
                st.markdown(
                    "<div style='color:#ee1d52; font-size:20px; font-weight:600; margin-top:6px;'>Sentiment</div>",
                    unsafe_allow_html=True
                )
                st.caption("Share of comments that are positive, negative or neutral")

                # Prepare data for sentiment pie chart
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

                # Create pie chart using Plotly
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

                # Display chart using unique key to avoid Streamlit element ID conflicts
                st.plotly_chart(fig, use_container_width=True, key=f"sentiment_pie_{row['Topic']}")

    # Add extra space after topic sections
    st.markdown("")
    st.markdown("")
