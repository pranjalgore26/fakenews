# Fake news checker (Flask + scikit-learn)

Paste an article, get a "leans fake / leans real" score plus the words that drove it.

## Run it

```bash
pip install -r requirements.txt

# 1. Get data: download the "Fake and Real News Dataset" (ISOT) from Kaggle
#    and put Fake.csv and True.csv in the data/ folder.
#    (Or provide data/news.csv with columns: text, label  -- 1 = fake, 0 = real)

python train.py     # trains and writes model.joblib
python app.py       # open http://127.0.0.1:5000
```

Without data files, `train.py` falls back to a tiny demo set so the site runs,
but that model is not accurate.

## How it works

- `textutils.py`: cleans text (removes "(Reuters)" datelines and URLs that would leak the label)
- `train.py`: TF-IDF (1-2 word n-grams) + logistic regression, prints held-out accuracy
- `app.py`: Flask server, `POST /predict` returns probability, verdict and top signal words
- `templates/index.html`: the page

## Limits

The model learns writing style from one dataset (mostly US political news, 2016-2017).
It does not check facts and will be less reliable on other topics, languages or eras.
Next steps: train on more varied data (e.g. LIAR, FakeNewsNet), try a transformer
such as DistilBERT, and add source/URL checks.
