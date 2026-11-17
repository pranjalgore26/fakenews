"""Train the fake-news classifier and save it to model.joblib.

Real data (recommended), either of:
  data/Fake.csv + data/True.csv   (ISOT / Kaggle "Fake and Real News" set)
  data/news.csv                   (columns: text, label  where 1 = fake, 0 = real)

With no data files, a tiny built-in demo set is used so the site still runs.
That demo model is NOT accurate; it only proves the pipeline works.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from textutils import clean

DATA = Path("data")
MODEL_PATH = Path("model.joblib")

DEMO_FAKE = [
    "SHOCKING: Doctors HATE this one weird trick the government is hiding from you, share before it gets deleted!",
    "BREAKING!!! Secret elite cabal caught rigging everything, mainstream media refuses to tell you the truth!",
    "You won't believe what this celebrity did, the media is silencing the truth about miracle cures big pharma fears!",
    "Wake up sheeple, the globalists are poisoning the water and they are laughing at you, share this now!",
    "Anonymous insider EXPOSES the hoax that the liars in power do not want you to read, patriots must act!",
    "Miracle fruit cures every disease overnight and scientists are terrified, click to see the banned video!",
    "The corrupt media is lying again, a whistleblower reveals the shocking cover-up they tried to destroy!",
    "Outrage as radical traitors plot to steal your freedom, forward this to everyone you know immediately!",
]
DEMO_REAL = [
    "The central bank held interest rates steady on Wednesday, citing moderate growth and easing inflation in the latest data.",
    "Lawmakers passed the transport funding bill by a vote of 62 to 38 after two days of debate, officials said on Thursday.",
    "The city council approved a new budget that raises spending on public schools by three percent, according to a statement.",
    "Researchers published a study in a peer-reviewed journal showing a modest reduction in symptoms among trial participants.",
    "The company reported quarterly revenue of 4.2 billion dollars, slightly above analyst expectations, in a filing on Tuesday.",
    "Officials said the bridge will close for repairs for six weeks, and traffic will be diverted along two alternate routes.",
    "The ministry announced new trade talks scheduled for next month, and a spokesperson said details would follow.",
    "Heavy rain caused flooding in several districts on Monday, and emergency services evacuated about 200 residents.",
]


def load_data() -> tuple[pd.DataFrame, bool]:
    fake_csv, true_csv, news_csv = DATA / "Fake.csv", DATA / "True.csv", DATA / "news.csv"

    if fake_csv.exists() and true_csv.exists():
        fake, real = pd.read_csv(fake_csv), pd.read_csv(true_csv)
        fake["label"], real["label"] = 1, 0
        df = pd.concat([fake, real], ignore_index=True)
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        return df[["text", "label"]], True

    if news_csv.exists():
        df = pd.read_csv(news_csv)[["text", "label"]].dropna()
        return df, True

    df = pd.DataFrame({
        "text": DEMO_FAKE + DEMO_REAL,
        "label": [1] * len(DEMO_FAKE) + [0] * len(DEMO_REAL),
    })
    return df, False


def main() -> None:
    df, real_data = load_data()
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.split().str.len() > 5].drop_duplicates("text")
    print(f"Training on {len(df):,} articles ({df['label'].mean():.0%} fake)")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            min_df=2 if real_data else 1, max_df=0.8, sublinear_tf=True,
            max_features=200_000,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=3.0)),
    ])

    if real_data:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"Held-out accuracy: {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, target_names=["real", "fake"]))
    else:
        pipe.fit(df["text"], df["label"])
        print("WARNING: no dataset found in data/, trained on a tiny demo set. "
              "Add Fake.csv + True.csv for a usable model.")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
