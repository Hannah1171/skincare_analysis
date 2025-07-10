import pandas as pd
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
    texts = df[text_col].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**encodings)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        for score in scores:
            label_idx = torch.argmax(score).item()
            confidence = score[label_idx].item()
            label = model.config.id2label[label_idx]
            results.append({sentiment_col: label, confidence_col: confidence})

    sentiment_df = pd.DataFrame(results)
    df = df.reset_index(drop=True)
    df[[sentiment_col, confidence_col]] = sentiment_df
    df.to_csv(output_path, index=False)

    print(f"Saved sentiment results to {output_path}")
    return df
