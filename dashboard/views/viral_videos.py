# views/viral_videos.py
import streamlit as st

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
