from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Data" / "dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

EXPECTED_COLUMNS = ["id", "entity", "sentiment", "text"]
VALID_SENTIMENTS = {"positive", "negative", "neutral", "irrelevant"}

def load_dataset(path=DATASET_PATH):
    """Load the repository dataset, which has no header row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put dataset.csv inside the project's Data folder."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=EXPECTED_COLUMNS,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )
    # Guard against a duplicate header if a user later replaces the file with a headed CSV.
    if not df.empty and str(df.iloc[0]["id"]).strip().lower() == "id":
        df = df.iloc[1:].copy()
    df["entity"] = df["entity"].fillna("Unknown").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["sentiment"].isin(VALID_SENTIMENTS)]
    df = df[df["text"].str.len() > 0]
    df["sentiment"] = df["sentiment"].str.title()
    return df.reset_index(drop=True)

def normalize_text(text):
    """Lightweight, consistent normalization shared by training and inference."""
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
