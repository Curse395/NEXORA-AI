"""
NEXORA — Feature engineering: CountVectorizer & TF-IDF.

These classical NLP features are used by the *auxiliary* language-identification
task (Naïve Bayes / Logistic Regression / SVM). The neural translation model
does NOT use these features — it learns its own subword embeddings inside the
Seq2Seq encoder. Using TF-IDF as classical features is standard, simple and
explainable, which is exactly what a classical-NLP baseline requires.

Why they are useful for classical models
----------------------------------------
* CountVectorizer builds a Bag-of-Words (BoW) matrix  X[i][j] = count of
  word j in document i. It is fast, sparse-friendly and captures which words
  appear, but it ignores rarity — frequent function words dominate.
* TF-IDF = Term Frequency × Inverse Document Frequency

      tfidf(t, d) = tf(t, d) * log( (1 + N) / (1 + df(t)) ) + 1

  It down-weights words that appear in almost every document and boosts words
  that are discriminative for a small subset of documents — ideal for
  *language identification*, where script-unique tokens are extremely
  informative (e.g. "है" for Hindi, "आहे" for Marathi, "भाषा" shared by both).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def char_analyzer(text: str) -> list[str]:
    """Character n-gram analyzer (n=2..4) — robust across scripts."""
    grams = []
    for n in (2, 3, 4):
        for i in range(max(len(text) - n + 1, 1)):
            grams.append(text[i : i + n])
    return grams


def build_features(
    df: pd.DataFrame,
    text_col: str = "text",
    vocab_size: int = 30_000,
    save_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """
    Build train/valid/test matrices for the 4 feature variants:

      count_vectorizer   — plain Bag-of-Words counts
      tfidf_word         — word-level TF-IDF
      tfidf_char         — character n-gram TF-IDF (default feature for the
                           classical models because script fingerprints live
                           in character n-grams)
      count_char         — character n-gram counts (used for the comparison)

    Returns the fitted vectorizers, the matrix split indices and the arrays.
    """
    from sklearn.model_selection import train_test_split

    X_text = df[text_col].fillna("").astype(str).tolist()
    y = df["language"].values

    # reproducible split (80/10/10)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_text, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    spec = {
        "count_vectorizer": {
            "vectorizer": CountVectorizer(max_features=vocab_size, ngram_range=(1, 1)),
            "label": "CountVectorizer (BoW)",
        },
        "tfidf_word": {
            "vectorizer": TfidfVectorizer(max_features=vocab_size, ngram_range=(1, 2)),
            "label": "TF-IDF (word n-grams)",
        },
        "count_char": {
            "vectorizer": CountVectorizer(max_features=vocab_size, analyzer=char_analyzer),
            "label": "CountVectorizer (char n-grams)",
        },
        "tfidf_char": {
            "vectorizer": TfidfVectorizer(max_features=vocab_size, analyzer=char_analyzer),
            "label": "TF-IDF (char n-grams) — default",
        },
    }

    fitted = {}
    for name, cfg in spec.items():
        vec = cfg["vectorizer"].fit(X_tr)
        fitted[name] = {
            "vectorizer": vec,
            "label": cfg["label"],
            "X_train": vec.transform(X_tr),
            "X_valid": vec.transform(X_va),
            "X_test": vec.transform(X_te),
            "feature_names": np.array(vec.get_feature_names_out()),
        }

    out = {
        "y_train": y_tr,
        "y_valid": y_va,
        "y_test": y_te,
        "fitted": fitted,
        "classes": sorted(set(y)),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, cfg in fitted.items():
            np.savez(
                save_dir / f"features_{name}.npz",
                X_train=cfg["X_train"], X_valid=cfg["X_valid"], X_test=cfg["X_test"],
                feature_names=cfg["feature_names"], classes=np.array(out["classes"]),
                y_train=out["y_train"], y_valid=out["y_valid"], y_test=out["y_test"],
            )
        print(f"[features] wrote feature matrices to {save_dir}")
    return out


def feature_overview(fitted: dict) -> pd.DataFrame:
    """Small summary table of the four feature spaces."""
    rows = []
    for name, cfg in fitted.items():
        rows.append({
            "Feature space": cfg["label"],
            "Vocabulary size": int(cfg["feature_names"].shape[0]),
            "Train matrix shape": f"{cfg['X_train'].shape}",
            "Sparsity %": round(100 * (1 - cfg["X_train"].nnz / (cfg["X_train"].shape[0] * cfg["X_train"].shape[1])), 2),
        })
    return pd.DataFrame(rows)