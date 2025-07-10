# Imports
from skincare.pipeline.preprocessing import get_stopwords, clean_text
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import stopwordsiso as stopwordsiso
import pandas as pd
import re
from typing import List
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


# Load LLM for topic name generation
llm = pipeline("text2text-generation", model="google/flan-t5-large")

# Sentence embedding model for label deduplication
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalize topic labels for deduplication
def normalize_label(label: str) -> str:
    return re.sub(r"[^\w\s]", "", label.lower()).strip()

def compute_topic_coherence(model, texts, topn=10):
    if isinstance(texts, pd.DataFrame):
        texts = texts["comment"].astype(str).apply(str.split).tolist()
    else:
        texts = [t.split() for t in texts]

    dictionary = Dictionary(texts)
    topics = [
        [w for w, _ in model.get_topic(tid)[:topn]]
        for tid in model.get_topics().keys() if tid != -1
    ]

    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    scores = cm.get_coherence_per_topic()
    return dict(zip([tid for tid in model.get_topics() if tid != -1], scores))


# Deduplicate similar topic labels using cosine similarity
def deduplicate_labels(topic_df, threshold=0.9):
    topic_df["NormName"] = topic_df["Name"].apply(normalize_label)
    embeddings = embedding_model.encode(topic_df["NormName"].tolist(), normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)
    label_map = {}
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] > threshold:
                label_map[topic_df.loc[j, "Name"]] = topic_df.loc[i, "Name"]
    topic_df["Name"] = topic_df["Name"].replace(label_map)
    return topic_df

# Filter topic keywords for LLM label generation
def get_filtered_keywords(model, topic_id, common_words, top_n=5):
    topic_words = model.get_topic(topic_id)
    filtered = [w for w, _ in topic_words if w not in common_words]
    return " ".join(filtered[:top_n])

# Generate a topic label using keywords and representative comments
def generate_topic_label(keywords: str, examples: List[str]) -> str:
    prompt = (
            "You are labeling clusters of social media skincare comments.\n"
            "Your task is to generate a short, clear topic name (1–3 words max) that summarizes the main theme of the entire cluster.\n"
            "Use noun phrases only (e.g., 'Sunscreen', 'Product reviews'). Avoid vague terms like 'Love' or 'Like' or full sentences.\n"
            "Focus on specific skincare concerns, techniques, or product types that describe the cluster as a whole.\n"
            "Generalize across all examples — do not base the label on just one comment.\n"
            "Examples:\n"
            "- Double cleansing\n"
            "- Tight budget\n"
            "- Hair\n"
            "- Sunscreen\n"
            "- Nose taping\n"
            "- Product reviews\n"
            "- Mask\n"
            "- Glow-up\n"
            "- Vitamin C\n"
            f"Top keywords: {keywords}\n"
            f"Representative comments:\n" + "\n".join(f"- {c}" for c in examples) + "\nTopic name:"
        )
    return llm(prompt)[0]["generated_text"].strip()

# Build and return a BERTopic model
def build_topic_model(
    embedding_model= embedding_model,
    stopwords=None,
    min_cluster_size=20,
    min_samples=10,
    language="english",
    use_umap=True,
    ngram_range=(1, 3),
):
    if stopwords is None:
        stopwords = get_stopwords(langs=["en"])

    vectorizer_model = CountVectorizer(stop_words=stopwords, ngram_range=ngram_range, max_features=3000)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=80,
        metric="euclidean",  # cosine or euclidean
        cluster_selection_method="eom",
        prediction_data=True
    )

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        random_state=36,
    ) if use_umap else None

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        language=language,
        calculate_probabilities=False,
        verbose=True
    )

# Build topic metadata including LLM labels and keyword filtering
def build_topic_metadata(df: pd.DataFrame, model, prob_col: str = "probability", include_time: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    stopwords = get_stopwords(langs=["en"])
    common_words = set(stopwords) | {"skin", "use", "product", "care", "good", "thing", "help"}
    summaries, result_rows = [], []
    has_comment, has_words = "comment" in df.columns, "Words" in df.columns
    has_timestamp = "timestamp" in df.columns and include_time

    for topic_id in sorted(df["topic"].unique()):
        subset = df[df["topic"] == topic_id] #.sort_values(prob_col, ascending=False)

        if topic_id == -1:
            label, examples, keywords = "Miscellaneous / Noise", [], ""
        else:
            examples = subset["comment"].head(4).tolist() if has_comment else []
            keywords = get_filtered_keywords(model, topic_id, common_words) if has_words and not subset.empty else ""
            label = generate_topic_label(keywords, examples)

        summaries.append({
            "Topic": topic_id,
            "TopicName": label,
            "Keywords": keywords,
            "ExampleComments": examples,
            "Time": f"{subset['timestamp'].min()} – {subset['timestamp'].max()}" if has_timestamp and not subset.empty else None
        })

        result_rows.extend([{**row, "TopicName": label, "TopicExamples": examples} for row in subset.to_dict(orient="records")])

    return pd.DataFrame(result_rows), pd.DataFrame(summaries)


# Run and label topics for a cleaned DataFrame
def run_topic_model(
    df: pd.DataFrame,
    text_col: str = "comment",
    embedding_model=embedding_model,
    min_cluster_size: int = 20,
    min_samples: int = 5
):
    df = df[df["textLanguage"] == "en"].copy()
    df[text_col] = df[text_col].astype(str)

    # Remove rows containing any unwanted keywords (case-insensitive)
    pattern = r"sagajewels|jewelry|collaboration|collab"
    df = df[~df[text_col].str.contains(pattern, case=False, na=False)]

    # Save original comment
    df["original_text"] = df[text_col]

    # Clean comments
    stopwords = get_stopwords(langs=["en"])
    df["cleaned_text"] = df[text_col].apply(lambda x: clean_text(x, stopwords=stopwords))

    # Filter and deduplicate
    df_filtered = df[df["cleaned_text"].str.split().str.len() > 5]
    df_unique = df_filtered.drop_duplicates(subset=["cleaned_text"])

    # Extract cleaned texts
    comments = df_unique["cleaned_text"].tolist()

    # Build and fit topic model
    model = build_topic_model(
        embedding_model=embedding_model,
        stopwords=stopwords,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        language="english"
    )
    
    topics, _ = model.fit_transform(comments)

    # Coherence filtering
    coherence_scores = compute_topic_coherence(model, df_unique)
    keep = {t for t, s in coherence_scores.items() if s >= 0.35}
    print(f"Dropped {len(coherence_scores) - len(keep)} low-coherence topics")

    df_unique["Topic"] = [t if t in keep else -1 for t in topics]
    df_unique["Words"] = df_unique["Topic"].map(
        lambda t: " ".join(w for w, _ in model.get_topic(t)) if t != -1 else ""
    )

    # Merge on cleaned_text to keep original_text
    df_labeled = df_filtered.merge(
        df_unique[["cleaned_text", "Topic", "Words", "original_text"]],
        on="cleaned_text",
        how="left"
    )

    # Topic statistics
    topic_counts = df_labeled["Topic"].value_counts().to_dict()
    examples = model.get_representative_docs()
    cleaned_to_original = dict(zip(df_unique["cleaned_text"], df_unique["original_text"]))

    valid_topic_ids = df_unique["Topic"].unique()

    # Summary per topic
    topic_summary = []
    for topic_id in sorted(valid_topic_ids):
        words = model.get_topic(topic_id)
        keywords = " ".join(w for w, _ in words)
        examples_cleaned = examples.get(topic_id, [])[:3]
        sample_comments = [cleaned_to_original.get(text, text) for text in examples_cleaned]
        label = generate_topic_label(keywords, examples_cleaned)
        topic_summary.append({
            "Topic": topic_id,
            "Name": label,
            "Count": topic_counts.get(topic_id, 0),
            "Keywords": keywords,
            "Examples": sample_comments
        })

    topic_summary = pd.DataFrame(topic_summary)
    topic_summary = deduplicate_labels(topic_summary, threshold=0.9)

    # Compute sentiment distribution per topic (excluding -1 and NaNs)
    valid_sentiment = df_labeled[df_labeled["Topic"].notna() & (df_labeled["Topic"] != -1)].copy()

    # Count sentiments per topic and normalize to get shares
    sentiment_dist = (
        valid_sentiment.groupby(["Topic", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum(), axis=1)
        .rename(columns={
            "positive": "positive_share",
            "negative": "negative_share",
            "neutral": "neutral_share"
        })
        .reset_index()
    )

    topic_summary = topic_summary.merge(sentiment_dist, on="Topic", how="left")

    # Merge names and counts back to full labeled data
    df_named = df_labeled.merge(
        topic_summary[["Topic", "Name", "Count"]],
        on="Topic",
        how="left"
    )

    return model, topic_summary, df_named


# Run hierarchical topic modeling (broad + subtopic levels)
def generate_hierarchical_topics(df: pd.DataFrame, text_col: str = "comment") -> pd.DataFrame:
    df[text_col] = df[text_col].astype(str)
    df_filtered = df[df[text_col].str.split().str.len() > 5]
    df_unique = df_filtered.drop_duplicates(subset=[text_col])
    documents = df_unique[text_col].tolist()
    stopwords = get_stopwords(langs=["en"])

    broad_model = build_topic_model(stopwords=stopwords, min_cluster_size=70, min_samples=10)
    broad_topics, _ = broad_model.fit_transform(documents)
    df_unique["SuperTopicIndex"] = broad_topics

    results = []
    for super_topic in sorted(df_unique["SuperTopicIndex"].unique()):
        if super_topic == -1:
            continue

        sub_df = df_unique[df_unique["SuperTopicIndex"] == super_topic]
        sub_comments = sub_df[text_col].tolist()

        sub_model = build_topic_model(stopwords=stopwords, min_cluster_size=15, min_samples=5)
        sub_topics, _ = sub_model.fit_transform(sub_comments)

        super_keywords_str = " ".join(w for w, _ in broad_model.get_topics().get(super_topic, []))
        super_label = generate_topic_label(super_keywords_str, sub_comments[:3])

        for sub_id in sorted(set(sub_topics)):
            if sub_id == -1:
                continue

            specific_comments = [c for i, c in enumerate(sub_comments) if sub_topics[i] == sub_id]
            sub_keywords_str = " ".join(w for w, _ in sub_model.get_topics().get(sub_id, []))
            sub_label = generate_topic_label(sub_keywords_str, specific_comments[:3])

            results.append({
                "SuperTopicIndex": super_topic,
                "SuperTopicName": super_label,
                "SubTopicIndex": sub_id,
                "SubTopicName": sub_label,
                "Comments": specific_comments[:3],
                "Frequency": len(specific_comments)
            })

    return pd.DataFrame(results)
