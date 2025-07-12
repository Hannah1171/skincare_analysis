import streamlit as st
import plotly.graph_objects as go
from data_loader import TIKTOK_PINK, POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR
import pandas as pd 

def display_ingredient_sentiment_ui(df_sentiments, df_examples):
    st.header("\U0001F9EA Ingredients Gen Z talks about")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("Analysis of how TikTok users feel about skincare ingredients.")

    df_sentiment = df_sentiments.copy()
    df_comments = df_examples.copy()
    top_ingredients = df_sentiment.sort_values("total_mentions", ascending=False).head(10)

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(top_ingredients.iterrows()):
        col = col1 if i % 2 == 0 else col2
        with col:
            with st.expander(f"\U0001F50D {row['matched_ingredients']} ({int(row['total_mentions'])} mentions)"):
                percent_width = min(row["total_mentions"] / df_sentiment["total_mentions"].max(), 1.0) * 100
                st.markdown(
                    f"""
                    <div style='background-color:#f5c0d1; border-radius:6px; height:16px; width:100%; margin-bottom:10px'>
                        <div style='background-color:{TIKTOK_PINK}; width:{percent_width:.2f}%; height:100%; border-radius:6px'></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

                st.markdown("#### Sentiment")
                st.caption("Share of comments that are positive, neutral or negative")
                labels = ["Positive", "Neutral", "Negative"]
                values = [row.get("positive", 0), row.get("neutral", 0), row.get("negative", 0)]
                custom_labels = [f"<b>{label}</b><br>{value:.0f}%" for label, value in zip(labels, values)]

                fig = go.Figure(data=[
                    go.Pie(
                        labels=custom_labels,
                        values=values,
                        marker=dict(colors=[POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR]),
                        textinfo="label",
                        hoverinfo="label+percent",
                        textfont=dict(size=14, color="white")
                    )
                ])
                fig.update_layout(margin=dict(t=10, b=10), showlegend=False, height=280)
                st.plotly_chart(fig, use_container_width=True)


def display_brand_sentiment_ui(df_sentiment, df_examples):
    st.subheader("Explore your Competitors")
    st.markdown("Brand-level sentiment breakdown from TikTok influencer captions.")

    df_sentiment.columns = df_sentiment.columns.str.lower()
    df_examples.columns = df_examples.columns.str.lower()

    available_brands = df_sentiment["matched_brand"].unique()
    selected_brand = st.selectbox("Choose a brand to explore", options=available_brands)
    row = df_sentiment[df_sentiment["matched_brand"] == selected_brand].iloc[0]

    st.markdown(f"#### Sentiment breakdown for **{selected_brand.title()}**")
    sentiment_labels = ["Positive", "Neutral", "Negative"]
    sentiment_values = [row.get("positive", 0), row.get("neutral", 0), row.get("negative", 0)]
    sentiment_percentages = [
        f"<b>{label}</b><br>{int(100 * value)}%" 
        for label, value in zip(sentiment_labels, row[["pct_positive", "pct_neutral", "pct_negative"]])
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

    st.markdown(f"#### Voice of Gen Z Influencers about **{selected_brand.title()}**")
    st.caption("Real Captions from TikTok Influencers")
    top_comments = (
        df_examples[df_examples["brand"] == selected_brand]
        .drop_duplicates("text")
        .sort_values("sentiment")
        .groupby("sentiment")
        .head(1)
    )

    for _, comment_row in top_comments.iterrows():
        st.markdown(
            f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&ldquo;&nbsp;</span>"
            f"<span style='font-size:15px; color:black;'>{comment_row['text'].strip()}</span>"
            f"<span style='color:{TIKTOK_PINK}; font-size:20px; font-weight:bold;'>&nbsp;&rdquo;</span>",
            unsafe_allow_html=True
        )

        if pd.notna(comment_row.get("bucketurl", "")):
            st.markdown(f"""
            <div style="text-align: center;">
                <iframe src="{comment_row['bucketurl']}" width="250" height="400" style="border:none;" allow="autoplay; fullscreen"></iframe>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")