import streamlit as st

def display_home():
    st.image("data/static_data/logo.png", use_container_width=True)
    st.title("✨ Welcome to the Skincare Dashboard")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("The data used for the dashboard consists of the last 9 months and was collected by scraping TikTok using the following hashtags #skincare, #skincareroutine, #hautpflege, #hautpflegeroutine.")
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Posts each month", "1.500")
    col2.metric("Comments (EN and DE) each month", "7000")
    col3.metric("Updated", "Weekly") 
    st.divider()
    st.markdown("### 🧑‍💻 How does GenZ talk about skincare?")

    st.markdown("By analyzing the most frequent word combinations (n-grams) in TikTok comments, we surfaced common phrases GenZ uses when talking about skincare. "
                "These phrases reflect curiosity, appreciation, and a desire to connect, with both products and creators.")

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
            <tr><th>Theme</th><th>Insight</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>Self-expression</strong></td><td>Discovery & connection</td></tr>
            <tr><td><strong>TikTok's Role</strong></td><td>Inspiration source & beauty search engine</td></tr>
            <tr><td><strong>Product Queries</strong></td><td>“Where to buy?” / “What’s the name?”</td></tr>
            <tr><td><strong>Comment Tone</strong></td><td>Curious, emotional; fosters engagement</td></tr>
            <tr><td><strong>Affirmations</strong></td><td>“love this”, “thank you” to connect with creators</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
