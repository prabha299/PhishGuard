import re
import string
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "phishing_email.csv"
MODEL_PATH = "phishing_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def clean_text(text: str) -> str:
    """Basic normalization: lowercase, strip URLs/punctuation/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)          # collapse links to a token
    text = re.sub(r"\S+@\S+", " EMAIL ", text)                  # collapse emails to a token
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be forgiving about column naming
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("email_text") or cols.get("body") or df.columns[0]
    label_col = cols.get("label") or cols.get("class") or cols.get("target") or df.columns[1]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])

    # Normalize label to 0/1 if given as strings like "phishing"/"legitimate"
    if df["label"].dtype == object:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .map({"phishing": 1, "spam": 1, "1": 1, "legitimate": 0, "ham": 0, "safe": 0, "0": 0})
        )
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    return df


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows ({df['label'].sum()} phishing / {(df['label'] == 0).sum()} legitimate)")

    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training LogisticRegression classifier ...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
