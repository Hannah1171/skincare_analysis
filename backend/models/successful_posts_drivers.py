import pandas as pd
import numpy as np
import re
import shap
from sklearn.model_selection import train_test_split
import spacy
import emoji
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
stopwords = STOP_WORDS

nlp = spacy.load("en_core_web_sm")  # English small model

def successful_posts_drivers(input_path: str):
    #raw_transcripts = pd.read_csv("../data/raw_transcripts.csv", encoding='utf-8')
    df = pd.read_csv(input_path, encoding='utf-8')
    df = get_data(df = df)
    df, df_hashtag_features =  prepare_for_model(df = df)
    df_clean, X_combined, additional_cols_final, vectorizer, additional_data_scaled, scaler = combine_features(df = df, df_hashtag_features=df_hashtag_features)
    relevant_features, shap_values, df_test_orig, test_idx = regression_model(df=df_clean, X_combined=X_combined, additional_cols_final=additional_cols_final,  vectorizer=vectorizer, additional_data_scaled=additional_data_scaled, scaler=scaler )
    get_shap_values(shap_values=shap_values, df_test_orig=df_test_orig, test_idx=test_idx)
    classification_model_is_viral(df=df, X_combined=X_combined)
    optimal_feature_range(shap_values, df_test_orig, relevant_features)
    
def get_data(df:pd.DataFrame):
    df['createTimeISO'] = (
        pd.to_datetime(df['createTimeISO'], utc=True, errors='coerce')
        .dt.tz_convert('Europe/Berlin')
    )

    nine_months = pd.Timestamp.utcnow() - pd.DateOffset(months=6)
    df = df[df['createTimeISO'] >= nine_months]
    df = df[df['textLanguage'] == 'en']

    # Create the concatenated text column
    text_cols = ['text', 'video_description', 'transcribed_text']
    df['text_all'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    # All three fields
    #df['processed_text'] = df.apply(lambda row: combine_text(row, include_description=True), axis=1)

    # Or, only caption + transcript
    df['combined_text'] = df.apply(lambda row: combine_text(row, include_description=False), axis=1)

    df['combined_text_processed'] = df['combined_text'].apply(lambda x: clean_text(x, stopwords=stopwords, lemmatize=True))

    # Get time based categories
    # Hour of posting (0–23)
    df['hour_posting'] = df['createTimeISO'].dt.hour

    # Day of week (0=Monday, …, 6=Sunday)
    df['day_of_week'] = df['createTimeISO'].dt.dayofweek

    # Is weekend flag (True if Saturday or Sunday)
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    df['time_period'] = df['hour_posting'].apply(categorize_period)

    # Extract text statistics for all text fields
    text_stats_columns = ['word_count', 'sentence_count', 
                        'avg_word_length', 'exclamation_count', 'question_count']

    df[text_stats_columns] = df['combined_text_processed'].apply(extract_text_statistics)

    # Extract hashtags
    df['hashtag_list'] = df['text'].apply(extract_hashtags)
    return df


def prepare_for_model(df: pd.DataFrame):
    # Flatten all hashtags and count them
    all_hashtags = [tag for tags in df['hashtag_list'].dropna() for tag in (tags if isinstance(tags, list) else [])]
    hashtag_counts = Counter(all_hashtags)

    # Calculate the 40th percentile threshold (so we keep hashtags used more often than 40% of the others)
    counts = np.array(list(hashtag_counts.values()))
    threshold = np.percentile(counts, 99)  # "above 40th percentile" == 60th percentile and up

    # Build the set of most-used hashtags
    popular_hashtags = {tag for tag, count in hashtag_counts.items() if count >= threshold}

    # Keep only those hashtags for each post
    def filter_popular(tags):
        if not isinstance(tags, list):
            return []
        return [tag for tag in tags if tag in popular_hashtags]

    df['popular_hashtags'] = df['hashtag_list'].apply(filter_popular)

    # One-hot encode
    mlb = MultiLabelBinarizer()
    hashtag_features = mlb.fit_transform(df['popular_hashtags'])
    hashtag_feature_names = ['hashtag_' + h for h in mlb.classes_]
    df_hashtag_features = pd.DataFrame(hashtag_features, columns=hashtag_feature_names, index=df.index)

    return df, df_hashtag_features


def combine_features(df: pd.DataFrame, df_hashtag_features: pd.DataFrame):

    additional_cols = [
        'isAd',
        'author_fans',
        'video_duration',
        'isSponsored',
        'hour_posting', 'day_of_week', 'is_weekend', 'time_period',
        'word_count', 'sentence_count',
        'avg_word_length', 'exclamation_count', 'question_count'
    ]

    # 1. Drop rows with NaNs in additional features
    df_clean = df.dropna(subset=additional_cols)

    # 2. Reset index so everything lines up in order
    df_clean = df_clean.reset_index(drop=True)
    df_hashtag_features_clean = df_hashtag_features.iloc[df_clean.index].reset_index(drop=True)

    # 1. Drop rows with NaNs in additional features
    df_clean = df.dropna(subset=additional_cols).reset_index(drop=True)
    df_hashtag_features_clean = df_hashtag_features.iloc[df_clean.index].reset_index(drop=True)

    # 2. Drop outliers from df_clean
    low, high = 0.0, 1
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    mask = np.ones(len(df_clean), dtype=bool)

    for col in num_cols:
        p1, p99 = df_clean[col].quantile([low, high])
        mask &= df_clean[col].between(p1, p99)

    # Get remaining indices before reset
    remaining_idx = df_clean[mask].index

    # Apply mask to both DataFrames
    df_clean = df_clean.loc[remaining_idx].reset_index(drop=True)
    df_hashtag_features_clean = df_hashtag_features_clean.loc[remaining_idx].reset_index(drop=True)

    # 3. Concatenate hashtag features to cleaned DataFrame (indices now match)
    df_clean = pd.concat([df_clean, df_hashtag_features_clean], axis=1)

    # 4. Convert booleans to int
    for col in ['isAd', 'isSponsored', 'is_weekend']:
        df_clean[col] = df_clean[col].astype(int)

    # 5. One-hot encode 'time_period'
    df_clean = pd.get_dummies(df_clean, columns=['time_period'], drop_first=True)

    # 6. Prepare final list of feature columns (including hashtag columns)
    hashtag_cols = list(df_hashtag_features_clean.columns)
    additional_cols_final = (
        [col for col in additional_cols if col != 'time_period'] +
        [c for c in df_clean.columns if c.startswith('time_period_')] +
        hashtag_cols
    )

    # 7. Extract and standardize features
    additional_data = df_clean[additional_cols_final]
    scaler = StandardScaler()
    additional_data_scaled = scaler.fit_transform(additional_data)
    additional_sparse = csr_matrix(additional_data_scaled)

    X_tfidf, vectorizer = get_tfidf(df=df)

    # 8. Make sure your TF-IDF matrix is in the same row order as df_clean
    X_tfidf_clean = X_tfidf[df_clean.index]

    # If you built X_tfidf on a column of the original df, it matches df_clean row-for-row after .reset_index(drop=True)

    # 9. Combine all features
    X_combined = hstack([X_tfidf_clean, additional_sparse])

    return df_clean, X_combined, additional_cols_final, vectorizer, additional_data_scaled, scaler

def get_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english',  # or your own stopword list
    lowercase=False,       # Already lowercased during preprocessing
    analyzer='word',
    token_pattern=r'\b\w+\b'
    )

    X_tfidf = vectorizer.fit_transform(df['combined_text_processed'])

    return X_tfidf, vectorizer


def regression_model(df, X_combined, additional_cols_final, vectorizer, additional_data_scaled, scaler):
    # Ensure viral_score and is_viral columns exist in the passed df
    df = df.copy()
    df['viral_score'] = np.log1p(df['playCount'])

    y = df['viral_score']

    n_total = X_combined.shape[0]
    split_point = int(n_total * 0.8)

    # Assuming data is sorted chronologically (oldest first)
    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n_total)

    X_train = X_combined[train_idx]
    X_test  = X_combined[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method='hist'
    )

    xgb.fit(X_train, y_train)

    # Recover original (unscaled) values
    X_scaled = additional_data_scaled
    X_orig = X_scaled * scaler.scale_ + scaler.mean_

    df_orig = pd.DataFrame(X_orig, columns=additional_cols_final, index=df.index)

    df_train_orig = df_orig.loc[train_idx]
    df_test_orig = df_orig.loc[test_idx]

    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    feature_names = tfidf_feature_names + additional_cols_final

    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names, index=train_idx)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names, index=test_idx)

    explainer = shap.Explainer(xgb, X_train_df)
    shap_values = explainer(X_test_df)

    df_imp = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    #thresh = 0.1 * df_imp["mean_abs_shap"].max()
    #relevant = df_imp[df_imp["mean_abs_shap"] >= thresh]

    relevant = df_imp.head(4)


    relevant_features = pd.DataFrame(relevant)

    return relevant_features, shap_values, df_test_orig, test_idx



def get_shap_values(shap_values, df_test_orig, test_idx):
    top_features = ["author_fans", "video_duration", "word_count", "isAd"] # ADJUST IF NEEDED 

    for feature in top_features:
        # Get raw feature values from test set aligned with SHAP values
        feature_values = df_test_orig.loc[test_idx, feature].reset_index(drop=True)
        shap_val = shap_values[:, feature].values

        if feature == "isAd":
            # Binary feature → group by 0/1
            df_plot = pd.DataFrame({
                feature: feature_values,
                "shap_value": shap_val
            })

            group_means = df_plot.groupby(feature)["shap_value"].mean()
            output_df = group_means.reset_index().rename(columns={"shap_value": "mean_shap_value"})
            output_df.to_csv(f"data/dashboard/shap_vs_{feature}.csv", index=False)

        else:
            # Continuous feature → bin, filter, and plot
            p5, p95 = np.percentile(feature_values, [5, 95])
            mask = (feature_values >= p5) & (feature_values <= p95)

            filtered_vals = feature_values[mask]
            filtered_shap = shap_val[mask]

            df_plot = pd.DataFrame({
                feature: filtered_vals,
                "shap_value": filtered_shap
            })

            bin_edges = np.linspace(filtered_vals.min(), filtered_vals.max(), 20)
            df_plot["bin"] = pd.cut(df_plot[feature], bins=bin_edges, include_lowest=True)

            bin_means = df_plot.groupby("bin")["shap_value"].mean()
            bin_centers = df_plot.groupby("bin")[feature].mean()

            output_df = pd.DataFrame({
                f"{feature}_bin_center": bin_centers,
                "mean_shap_value": bin_means
            }).reset_index(drop=True)

            output_df.to_csv(f"data/dashboard/shap_vs_{feature}.csv", index=False)


def classification_model_is_viral(df, X_combined):
    df = df.copy()
    df = df.sort_values('createTimeISO').reset_index(drop=True)
    df['viral_score'] = np.log1p(df['playCount'])
    threshold = df['viral_score'].quantile(0.75)
    threshold = df['viral_score'].quantile(0.75)
    df['is_viral'] = (df['viral_score'] > threshold).astype(int)
    y = df['is_viral']

    n_total = X_combined.shape[0]
    split_point = int(n_total * 0.8)

    # Assuming data is sorted chronologically (oldest first)
    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n_total)


    X_train = X_combined[train_idx]
    X_test  = X_combined[test_idx]

    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    scale_pos_weight = 3.0 #0.75 / 0.25

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method='hist',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))

def optimal_feature_range(shap_values, df_test_orig, relevant_features):
    top_features = ["author_fans", "video_duration", "word_count", "isAd", "hour_posting"]

    # Filter relevant features first
    relevant_features = relevant_features[relevant_features["feature"].isin(top_features)].reset_index(drop=True)

    optimal_ranges = []

    for feat in relevant_features["feature"]:
        df_feat = pd.DataFrame({
            "feature_val": df_test_orig[feat].values,
            "shap_val": shap_values[:, feat].values
        })

        # Remove outliers
        p1, p99 = np.percentile(df_feat["feature_val"], [1, 99])
        df_feat = df_feat[(df_feat["feature_val"] >= p1) & (df_feat["feature_val"] <= p99)]

        # Bin + compute mean SHAP
        bins = pd.cut(df_feat["feature_val"], bins=100)
        summary = df_feat.groupby(bins)["shap_val"].mean()

        # Get bin with max SHAP value
        if not summary.empty:
            best_bin = summary.idxmax()
            best_range = f"{best_bin.left:.1f}–{best_bin.right:.1f}"
        else:
            best_range = "n/a"

        optimal_ranges.append(best_range)

    # Add the optimal ranges to the *filtered* relevant features
    relevant_features["optimal_value_range"] = optimal_ranges

    # Save to CSV
    relevant_features.to_csv("data/dashboard/successful_post_range.csv", index=False)


def combine_text(row, include_description=True):
    # Start with caption
    texts = []
    if pd.notna(row['text']):
        texts.append(str(row['text']))
    if pd.notna(row['transcribed_text']):
        texts.append(str(row['transcribed_text']))
    if include_description and pd.notna(row['video_description']):
        texts.append(str(row['video_description']))
    return ' '.join(texts)

def clean_text(text, stopwords=None, lemmatize=False):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = emoji.replace_emoji(text, replace=lambda e, _: f" {e} ")
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9\s#?!\U0001F600-\U0001F64F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Use spaCy tokenizer for more robust tokenization
    tokens = [token.text for token in nlp(text)]
    tokens = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in tokens]

    if stopwords:
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    else:
        tokens = [t for t in tokens if len(t) > 2]

    if lemmatize and tokens:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]

    return " ".join(tokens)


def pos_enhanced_preprocessing(text):
    meaningful_pos = {'NN', 'NNS', 'NNP', 'NNPS',
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                      'JJ', 'JJR', 'JJS',
                      'RB', 'RBR', 'RBS'}
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    meaningful_tokens = [token for token, pos in pos_tags if pos in meaningful_pos]
    return ' '.join(meaningful_tokens)


# Time period categories
def categorize_period(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def extract_text_statistics(text):
    """
    Extract statistical features from text content
    """
    if pd.isna(text) or text == '':
        return pd.Series([0, 0, 0, 0, 0])
    
    # Basic statistics
    word_count = len(str(text).split())
    sentence_count = len(str(text).split('.'))
    
    # Advanced statistics
    avg_word_length = np.mean([len(word) for word in str(text).split()]) if word_count > 0 else 0
    exclamation_count = str(text).count('!')
    question_count = str(text).count('?')
    
    return pd.Series([word_count, sentence_count, 
                     avg_word_length, exclamation_count, question_count])


def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    hashtags = re.findall(r'#\w+', text.lower())
    return [tag.replace('#', '') for tag in hashtags]



