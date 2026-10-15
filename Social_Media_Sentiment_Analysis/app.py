from pathlib import Path
import json
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from Analysis.data_utils import load_dataset, normalize_text, PROJECT_ROOT, DATASET_PATH, MODEL_PATH, METRICS_PATH

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-before-deploy")

MODEL = None
METRICS = {}


def load_artifacts():
    global MODEL, METRICS

    # Load the saved machine-learning model
    if MODEL_PATH.exists():
        try:
            MODEL = joblib.load(MODEL_PATH)
        except Exception as exc:
            print(f"Could not load model: {exc}")
            MODEL = None

    # Load real evaluation metrics
    METRICS = {}

    if METRICS_PATH.exists():
        try:
            METRICS = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not load metrics: {exc}")

    # Read the actual classification report
    report = METRICS.get("classification_report", {})
    macro = report.get("macro avg", {})

    # Use the real metrics.json values
    METRICS["accuracy_percent"] = round(
        float(METRICS.get("accuracy", 0)) * 100, 2
    )
    METRICS["macro_f1_percent"] = round(
        float(macro.get("f1-score", 0)) * 100, 2
    )
    METRICS["macro_precision_percent"] = round(
        float(macro.get("precision", 0)) * 100, 2
    )
    METRICS["macro_recall_percent"] = round(
        float(macro.get("recall", 0)) * 100, 2
    )

    # Match the dashboard's training/test row counts
    METRICS["training_rows"] = int(
        METRICS.get("training_records", 0)
    )
    METRICS["test_rows"] = int(
        METRICS.get("testing_records", 0)
    )

    print("Loaded model metrics:", METRICS)



def ensure_model():
    """Train the model once if saved artifacts do not exist yet."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            from Analysis.train_model import train_and_save
            train_and_save()
        MODEL = joblib.load(MODEL_PATH)

load_artifacts()

@app.route("/")
def dashboard():
    try:
        df = load_dataset()
        total = int(len(df))
        sentiment_counts = df["sentiment"].value_counts().to_dict()
        entity_counts = sorted(df["entity"].dropna().astype(str).unique().tolist())
        top_entities = (df.groupby("entity").size().sort_values(ascending=False).head(8)
                        .rename_axis("entity").reset_index(name="count").to_dict("records"))
        recent_rows = df[["entity", "sentiment", "text"]].dropna().head(8).to_dict("records")
    except Exception as exc:
        total, sentiment_counts, entity_counts, top_entities, recent_rows = 0, {}, [], [], []
        flash(f"Could not load dataset: {exc}", "error")
    return render_template(
        "index.html", total=total, sentiment_counts=sentiment_counts,
        entity_counts=entity_counts, top_entities=top_entities,
        recent_rows=recent_rows, metrics=METRICS
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    text_value = ""
    if request.method == "POST":
        text_value = request.form.get("text", "").strip()
        if not text_value:
            flash("Please enter a post or tweet first.", "error")
        else:
            try:
                ensure_model()
                cleaned_text = normalize_text(text_value)
                predicted = str(MODEL.predict([cleaned_text])[0])
                confidence = None
                if hasattr(MODEL, "predict_proba"):
                    probs = MODEL.predict_proba([cleaned_text])[0]
                    confidence = round(float(max(probs)) * 100, 1)
                result = {"label": predicted, "confidence": confidence}
            except Exception as exc:
                flash(f"Prediction failed: {exc}", "error")
    return render_template("predict.html", result=result, text_value=text_value)

@app.route("/dataset")
def dataset():
    try:
        df = load_dataset()
        entity = request.args.get("entity", "").strip()
        sentiment = request.args.get("sentiment", "").strip()
        query = request.args.get("q", "").strip()
        if entity:
            df = df[df["entity"].astype(str) == entity]
        if sentiment:
            df = df[df["sentiment"].astype(str).str.lower() == sentiment.lower()]
        if query:
            df = df[df["text"].astype(str).str.contains(query, case=False, na=False, regex=False)]
        total_filtered = len(df)
        page = max(1, request.args.get("page", 1, type=int))
        per_page = 25
        pages = max(1, (total_filtered + per_page - 1) // per_page)
        page = min(page, pages)
        rows = df.iloc[(page - 1) * per_page:page * per_page][["entity", "sentiment", "text"]].to_dict("records")
        entities = sorted(load_dataset()["entity"].dropna().astype(str).unique().tolist())
        sentiments = sorted(load_dataset()["sentiment"].dropna().astype(str).unique().tolist())
    except Exception as exc:
        flash(f"Could not display dataset: {exc}", "error")
        rows, entities, sentiments, total_filtered, page, pages = [], [], [], 0, 1, 1
        entity = sentiment = query = ""
    return render_template("dataset.html", rows=rows, entities=entities, sentiments=sentiments,
                           selected_entity=entity, selected_sentiment=sentiment, query=query,
                           total_filtered=total_filtered, page=page, pages=pages)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    # For local development only. Use a production WSGI server when deploying.
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
