from pathlib import Path
import os

import joblib
from flask import Flask, jsonify, render_template, request

from textutils import clean


# -------------------------------
# APPLICATION CONFIGURATION
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

MIN_WORDS = 25

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# -------------------------------
# LOAD MACHINE LEARNING MODEL
# -------------------------------

pipe = None

try:
    if MODEL_PATH.exists():
        pipe = joblib.load(MODEL_PATH)
        print("Machine Learning model loaded successfully!")
    else:
        print("WARNING: model.joblib not found!")

except Exception as e:
    print(f"Error loading model: {e}")


# -------------------------------
# EXPLAIN PREDICTION
# -------------------------------

def explain(cleaned: str, top_n: int = 6) -> dict:

    if pipe is None:
        return {"fake": [], "real": []}

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    names = tfidf.get_feature_names_out()

    x = tfidf.transform([cleaned]).multiply(
        clf.coef_[0]
    ).tocoo()

    pairs = sorted(
        zip(x.data, x.col),
        key=lambda p: p[0]
    )

    fake = [
        names[c]
        for w, c in reversed(pairs)
        if w > 0
    ][:top_n]

    real = [
        names[c]
        for w, c in pairs
        if w < 0
    ][:top_n]

    return {
        "fake": fake,
        "real": real
    }


# -------------------------------
# HOME PAGE
# -------------------------------

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html",
        min_words=MIN_WORDS
    )


# -------------------------------
# FAKE NEWS PREDICTION
# -------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if pipe is None:
        return jsonify({
            "error": "Machine learning model not found. Please check model.joblib."
        }), 503

    try:

        data = request.get_json(silent=True) or {}

        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({
                "error": "Invalid input. Please enter article text."
            }), 400

        text = text.strip()

        if not text:
            return jsonify({
                "error": "Please enter some news text."
            }), 400

        cleaned = clean(text)

        if len(cleaned.split()) < MIN_WORDS:
            return jsonify({
                "error": f"Please enter at least {MIN_WORDS} words for accurate analysis."
            }), 400

        # Get model classes
        classes = list(pipe.named_steps["clf"].classes_)

        if 1 not in classes:
            return jsonify({
                "error": "Model does not contain the expected fake-news class."
            }), 500

        # Predict probability
        probabilities = pipe.predict_proba([cleaned])[0]

        p_fake = float(
            probabilities[classes.index(1)]
        )

        # Determine verdict
        verdict = "fake" if p_fake >= 0.5 else "real"

        # Generate explanation
        signals = explain(cleaned)

        return jsonify({

            "fake_probability": round(p_fake, 4),

            "verdict": verdict,

            "signals": signals

        })

    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "error": "Something went wrong while analyzing the news."
        }), 500


# -------------------------------
# ERROR HANDLING
# -------------------------------

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "error": "Page not found."
    }), 404


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Input is too large. Please reduce the text length."
    }), 413


# -------------------------------
# APPLICATION START
# -------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
