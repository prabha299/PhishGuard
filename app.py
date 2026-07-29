import os
import re
import sqlite3
import string
from datetime import datetime

import joblib
from flask import Flask, g, render_template, request, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

app = Flask(__name__)

model = None
vectorizer = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as exc:  # noqa: BLE001
    MODEL_LOAD_ERROR = (
        "Model files not found or failed to load. "
        "Run `python train_model.py` first to generate "
        "phishing_model.pkl and vectorizer.pkl. "
        f"(details: {exc})"
    )


def clean_text(text: str) -> str:
    """Must mirror the preprocessing used in train_model.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text



def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_snippet TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_scan(email_text: str, prediction: str, confidence: float):
    snippet = email_text.strip()[:200]
    db = get_db()
    db.execute(
        "INSERT INTO scans (email_snippet, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
        (snippet, prediction, confidence, datetime.utcnow().isoformat(timespec="seconds")),
    )
    db.commit()


def get_recent_scans(limit: int = 10):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return rows



@app.route("/", methods=["GET"])
def index():
    history = get_recent_scans() if MODEL_LOAD_ERROR is None else []
    return render_template("index.html", error=MODEL_LOAD_ERROR, history=history)


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_LOAD_ERROR:
        return render_template("index.html", error=MODEL_LOAD_ERROR, history=[])

    email_text = request.form.get("email_text", "").strip()

    if not email_text:
        return render_template(
            "index.html",
            error="Please paste some email text before scanning.",
            history=get_recent_scans(),
        )

    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = float(proba[pred]) * 100

    label = "Phishing" if pred == 1 else "Legitimate"
    log_scan(email_text, label, confidence)

    return render_template(
        "result.html",
        original_text=email_text,
        prediction=label,
        confidence=round(confidence, 2),
        is_phishing=(pred == 1),
    )


@app.route("/history")
def history():
    return render_template("index.html", error=MODEL_LOAD_ERROR, history=get_recent_scans(limit=50))


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        init_db()
    else:
        init_db()  # ensure table exists even if file exists but is empty
    app.run(debug=True)
