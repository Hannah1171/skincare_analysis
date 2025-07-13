import streamlit as st
import plotly.graph_objects as go
from data_loader import TIKTOK_PINK, POSITIVE_FILL, NEGATIVE_FILL

def display_successful_post_insights(general, fans, hour, duration, wordcount, isad):
    st.header("\U0001F680 What makes a post go viral?")
    st.badge("DISCLAIMER", color='orange')
    st.markdown("Analysis of key post-level factors affecting virality on TikTok based on SHAP values.")

    sorted_df = general.sort_values("mean_abs_shap", ascending=True)
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
        text=[f"<b>&nbsp;&nbsp;{feat}</b>" for feat in labels],
        textposition="inside",
        insidetextanchor="start",
        insidetextfont=dict(color="white", size=14),
    ))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(title="Feature Drivers", showticklabels=False, showgrid=False),
        margin=dict(t=10, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Optimal settings for virality")
    general["optimal_value_range"] = general["optimal_value_range"].astype(str).str.replace(".0", "", regex=False)
    cols = st.columns(5)
    labels = ["Followers", "Posting Time", "Video Length", "Caption Word Count", "Ad"]
    values = ["> 500k", "18–23h", "20–30s", "16 words", "No ad"]
    for col, label, value in zip(cols, labels, values):
        col.metric(label=label, value=value)

    feature_map = {
        "Followers": fans,
        "Posting Time": hour,
        "Video Length": duration,
        "Caption Word Count": wordcount,
        "Ad": isad
    }
    label_to_feature = {
        "Followers": "author_fans",
        "Posting Time": "hour_posting",
        "Video Length": "video_duration",
        "Caption Word Count": "word_count",
        "Ad": "is_ad"
    }

    st.markdown("""
    <div style='font-size:20px; font-weight:600; color:#000; margin-bottom:0px;'>Select a driver to explore</div>
    """, unsafe_allow_html=True)
    selected_label = st.selectbox("", list(label_to_feature.keys()))
    selected_feature = label_to_feature[selected_label]
    df_plot = feature_map[selected_label]

    x_col, y_col = df_plot.columns
    x = df_plot[x_col].values
    y = df_plot[y_col].values

    fig = go.Figure()
    if selected_feature == "is_ad":
        colors = [POSITIVE_FILL if val >= 0 else NEGATIVE_FILL for val in y]
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
        fig.update_layout(
            xaxis_title="Ad (0 = No, 1 = Yes)",
            yaxis_title="Impact on Virality",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20)
        )
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=x, y=[val if val > 0 else 0 for val in y], fill='tozeroy', mode='none', fillcolor=POSITIVE_FILL))
        fig.add_trace(go.Scatter(x=x, y=[val if val < 0 else 0 for val in y], fill='tozeroy', mode='none', fillcolor=NEGATIVE_FILL))
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
