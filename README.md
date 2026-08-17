# Fake News Detection — Classical ML Demo

An educational Python project that demonstrates text-feature engineering and
classical machine-learning classification for news-like text. It is not a
production moderation system and must not be used to make automated content
or hiring decisions.

## What is implemented

- A 10-article, in-code demonstration dataset (five real-labelled and five
  fake-labelled examples).
- A separately committed CSV with 8 sample records.
- Text cleaning plus engineered text, source-credibility, bias, and clickbait
  features.
- Four scikit-learn classifiers: Logistic Regression, Random Forest, RBF SVM,
  and Multinomial Naive Bayes; a soft-voting ensemble is then trained from
  those classifiers.
- A 70/30 stratified demonstration split with `random_state=42` in the runnable
  demonstration path.

The project does **not** contain transformer models, external training data,
MLflow, DVC, SHAP/LIME explanations, FastAPI, or measured production metrics.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r fake_news_detection/requirements.txt
python -m fake_news_detection.main
```

Run tests with:

```bash
pytest -q
```

## Repository layout

```text
fake_news_detection/
  fake_news_detector.py  # feature engineering, training, evaluation, demo
  main.py                # runnable demonstration entry point
  web_app.py             # optional Streamlit interface
  data/sample_data.csv   # 8 manually committed sample records
tests/
  test_demo_data.py
```

## Limitations

The data is intentionally small and synthetic. Results from it are not useful
estimates of real-world misinformation detection performance. A future version
would need a documented, licensed dataset; held-out evaluation; bias analysis;
and reproducible experiment tracking.
