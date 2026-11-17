from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from textutils import clean

MODEL_PATH = Path("model.joblib")
MIN_WORDS = 25

app = Flask(__name__)
pipe = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def explain(cleaned: str, top_n: int = 6) -> dict:
    """Words in this article that pushed the score toward fake or real."""
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = tfidf.get_feature_names_out()
    x = tfidf.transform([cleaned]).multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(x.data, x.col), key=lambda p: p[0])
    fake = [names[c] for w, c in reversed(pairs) if w > 0][:top_n]
    real = [names[c] for w, c in pairs if w < 0][:top_n]
    return {"fake": fake, "real": real}


@app.get("/")
def index():
    return render_template("index.html", min_words=MIN_WORDS)


@app.post("/predict")
def predict():
    if pipe is None:
        return jsonify(error="No model found. Run `python train.py` first."), 503

    text = (request.get_json(silent=True) or {}).get("text", "")
    cleaned = clean(text)
    if len(cleaned.split()) < MIN_WORDS:
        return jsonify(error=f"Paste at least {MIN_WORDS} words. Short snippets are too noisy to score."), 400

    classes = list(pipe.named_steps["clf"].classes_)
    p_fake = float(pipe.predict_proba([cleaned])[0][classes.index(1)])
    return jsonify(
        fake_probability=round(p_fake, 4),
        verdict="fake" if p_fake >= 0.5 else "real",
        signals=explain(cleaned),
    )


if __name__ == "__main__":
    app.run(debug=True)
