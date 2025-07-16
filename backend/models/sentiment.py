import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_sentiment_analysis_on_comments(
    df: pd.DataFrame,
    output_path: str,
    text_col: str = "comment",
    sentiment_col: str = "sentiment",
    confidence_col: str = "sentiment_confidence",
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    batch_size: int = 32,
    max_length: int = 512
) -> pd.DataFrame:
    # Prepare text data: replace missing entries with empty strings and convert to list of strings
    texts = df[text_col].fillna("").astype(str).tolist()

    # Load tokenizer and sentiment model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode (disables dropout)

    results = []  # To store prediction results

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        # Run model inference without tracking gradients (saves memory)
        with torch.no_grad():
            outputs = model(**encodings)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)  # Convert logits to probabilities

        # Process each prediction in the batch
        for score in scores:
            label_idx = torch.argmax(score).item()                  # Class index with highest score
            confidence = score[label_idx].item()                    # Probability of the predicted class
            label = model.config.id2label[label_idx]                # Map index to label (e.g., POSITIVE, NEGATIVE)
            results.append({sentiment_col: label, confidence_col: confidence})

    # Create DataFrame from predictions
    sentiment_df = pd.DataFrame(results)

    # Filter out low-confidence predictions by setting sentiment to NaN
    sentiment_df.loc[sentiment_df[confidence_col] < 0.65, sentiment_col] = np.nan

    # Assign predictions back to original DataFrame
    df = df.reset_index(drop=True)
    df[[sentiment_col, confidence_col]] = sentiment_df

    # Save the updated DataFrame to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment results to {output_path}")

    return df  # Return DataFrame with sentiment predictions
