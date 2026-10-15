
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_dataset():
    """Load and clean the sentiment dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Dataset columns: id, entity, sentiment, text
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=["id", "entity", "sentiment", "text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    # Remove a possible header row
    df["sentiment"] = (
        df["sentiment"].astype(str).str.strip().str.lower()
    )

    valid_labels = [
        "positive",
        "negative",
        "neutral",
        "irrelevant",
    ]

    df = df[df["sentiment"].isin(valid_labels)].copy()

    # Clean text
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text", "sentiment"])

    if df["sentiment"].nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two sentiment classes."
        )

    return df


def main():
    print("Loading dataset...")
    df = load_dataset()

    X = df["text"]
    y = df["sentiment"]

    print(f"Total records: {len(df)}")
    print("Sentiment distribution:")
    print(y.value_counts())

    # Split data while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # lbfgs supports multiclass classification
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining sentiment model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "sentiment_classes": sorted(y.unique().tolist()),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=sorted(y.unique()),
        ).tolist(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    # Test different sentences
    examples = [
        "I absolutely love this product!",
        "This is the worst product ever.",
        "The product arrived today.",
        "I expected better quality for the price.",
    ]

    print("\nSample predictions:")
    for sentence in examples:
        result = model.predict([sentence])[0]
        print(f"Text: {sentence}")
        print(f"Predicted sentiment: {result}\n")


if __name__ == "__main__":
    main()
