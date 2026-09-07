"""
NEXORA — Classical machine-learning baselines for language identification.

Three models are trained on character n-gram TF-IDF features:

1. Multinomial Naïve Bayes   — generative, Bayes' rule with multinomial
                               likelihood; fast and strong on text.
2. Logistic Regression       — discriminative, L2-regularised, softmax
                               classification.
3. Linear Support Vector Machine (SVM) — max-margin linear classifier.

All results (accuracy, macro-F1, per-class precision/recall/F1, confusion
matrix, ROC AUC where available) are computed on the held-out test set and
saved under ``outputs/metrics/``. Nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --- consistent styling + Unicode fonts ------------------------------------
NEXORA_BLUE = "#2563eb"
NEXORA_PURPLE = "#7c3aed"
NEXORA_TEAL = "#0ea5e9"
NEXORA_ACCENT = "#f59e0b"


def _register_unicode_fonts() -> None:
    from matplotlib import font_manager

    for p in (r"C:\Windows\Fonts\Nirmala.ttf", r"C:\Windows\Fonts\Mangal.ttf",
              r"C:\Windows\Fonts\ArialUni.ttf"):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            continue
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = ["Nirmala UI", "Mangal", "Arial Unicode MS", "DejaVu Sans"]


_register_unicode_fonts()

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#1e293b",
    "text.color": "#1e293b",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

MODELS = {
    "naive_bayes": ("Multinomial Naïve Bayes", MultinomialNB(alpha=1.0)),
    "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=2000, C=1.0)),
    "svm": ("Support Vector Machine (LinearSVC)", LinearSVC(max_iter=20_000, C=1.0)),
}


def train_all(feats: dict, save_dir: Path | None = None) -> dict:
    """
    Train all three models on the default TF-IDF char feature space and
    evaluate them on the test split.

    Returns dict:
      results       — per-model metrics table
      reports       — per-model classification report (text)
      confusion     — {model: (confusion_matrix, classes)}
      roc           — {model: (macro_roc_auc, n_classes) or None}
      fitted        — the fitted sklearn estimators
    """
    default = feats["fitted"]["tfidf_char"]
    X_tr, y_tr = default["X_train"], feats["y_train"]
    X_va, y_va = default["X_valid"], feats["y_valid"]
    X_te, y_te = default["X_test"], feats["y_test"]
    classes = feats["classes"]

    results, reports, confusion, roc, fitted = {}, {}, {}, {}, {}
    for key, (label, model) in MODELS.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_te, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        results[key] = {
            "model": label,
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        }
        reports[key] = classification_report(y_te, y_pred, target_names=classes, zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=classes)
        confusion[key] = (cm, classes)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)
            try:
                roc[key] = float(roc_auc_score(y_te, proba, multi_class="ovr", labels=classes))
            except ValueError:
                roc[key] = None
        else:
            roc[key] = None
        fitted[key] = model

    results_df = pd.DataFrame(results).T.reset_index(drop=True)

    payload = {
        "results": results_df.to_dict(orient="records"),
        "reports": reports,
        "confusion": {k: (v[0].tolist(), v[1]) for k, v in confusion.items()},
        "roc": roc,
        "classes": classes,
    }
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_dir / "classical_results.csv", index=False)
        with open(save_dir / "classical_metrics.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[classical] wrote metrics to {save_dir}")
    return {"results": results_df, "reports": reports, "confusion": confusion,
            "roc": roc, "fitted": fitted}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(payload: dict, out_dir: Path) -> Path:
    """One confusion-matrix figure per model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (cm, classes) in payload["confusion"].items():
        cm = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), [c.capitalize() for c in classes], rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), [c.capitalize() for c in classes])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {MODELS[key][0]}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = out_dir / f"confusion_{key}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths[key] = p
    return paths


def plot_roc_curves(payload: dict, feats: dict, out_dir: Path) -> Path | None:
    """One-vs-rest ROC curve figure for models with predict_proba."""
    from sklearn.metrics import roc_curve

    X_te = feats["fitted"]["tfidf_char"]["X_test"]
    y_te = feats["y_test"]
    classes = feats["classes"]
    onevs = {c: (y_te == c).astype(int) for c in classes}

    fig, ax = plt.subplots(figsize=(9, 6))
    found = False
    for key in MODELS:
        roc_val = payload["roc"].get(key)
        if roc_val is None:
            continue
        found = True
        model = payload["fitted"][key]
        proba = model.predict_proba(X_te)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(onevs[c], proba[:, i])
            ax.plot(fpr, tpr, lw=1.2, alpha=0.55)
        # macro average line
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(onevs[classes[i]], proba[:, i])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        ax.plot(mean_fpr, mean_tpr, lw=2.4, label=f"{MODELS[key][0]} (macro AUC {roc_val:.3f})")
    if not found:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], ls="--", c="#94a3b8")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (macro average per model)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "roc_curves.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_model_comparison(payload: dict, out_dir: Path) -> Path:
    """Grouped bar chart comparing accuracy/precision/recall/F1."""
    results = payload["results"]
    keys = results["model"].tolist()
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(keys))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [NEXORA_BLUE, NEXORA_PURPLE, NEXORA_TEAL, NEXORA_ACCENT]
    for i, m in enumerate(metrics):
        vals = results[m].values
        ax.bar(x + (i - 1.5) * width, vals, width, label=m.replace("_", " ").title(), color=colors[i])
        for j, v in enumerate(vals):
            ax.text(j + (i - 1.5) * width, v + 0.005, f"{v:.2f}", fontsize=8, ha="center")
    ax.set_xticks(x, keys, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — language identification (test set)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / "model_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


NEXORA_BLUE = "#2563eb"
NEXORA_PURPLE = "#7c3aed"
NEXORA_TEAL = "#0ea5e9"