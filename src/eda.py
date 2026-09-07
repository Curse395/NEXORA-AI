"""
NEXORA — Exploratory Data Analysis.

All plots are generated from the *actual* processed dataset (no hard-coded
statistics). Figures are saved under ``outputs/plots/`` and mirrored on the
dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.multilingual import (
    detect_script,
    light_tokenize,
    remove_language_stopwords,
    word_frequencies,
)

# Consistent Nexora styling -------------------------------------------------
NEXORA_BLUE = "#2563eb"
NEXORA_PURPLE = "#7c3aed"
NEXORA_TEAL = "#0ea5e9"
NEXORA_ACCENT = "#f59e0b"
BG = "#ffffff"


def _register_unicode_fonts() -> None:
    """
    Register Windows fonts that cover Latin + Indic scripts (Devanagari,
    Bengali, Gujarati, Tamil, Telugu) so matplotlib labels render correctly
    instead of showing hollow boxes. Falls back to DejaVu silently.
    """
    from matplotlib import font_manager

    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",       # Nirmala UI (Indic + Latin)
        r"C:\Windows\Fonts\NirmalaB.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",        # Devanagari
        r"C:\Windows\Fonts\ArialUni.ttf",      # Arial Unicode MS
    ]
    available = []
    for p in candidates:
        try:
            font_manager.fontManager.addfont(p)
            available.append(p)
        except Exception:
            continue
    if available:
        import matplotlib as mpl

        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = ["Nirmala UI", "Mangal", "Arial Unicode MS", "DejaVu Sans"]


_register_unicode_fonts()

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
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


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(lid_df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Build a tidy "Nexora EDA statistics" table:
    rows = languages, columns = counts, vocabulary, mean/median/max lengths.
    """
    rows = []
    for lang in sorted(lid_df["language"].unique()):
        col = lid_df.loc[lid_df["language"] == lang, "text"]
        tokens = [t for text in col.tolist() for t in light_tokenize(text)]
        lengths = col.str.split().str.len()
        vocab = len(set(tokens))
        rows.append({
            "language": lang.capitalize(),
            "samples": int(len(col)),
            "tokens": len(tokens),
            "vocabulary": vocab,
            "mean_len": round(float(lengths.mean()), 2),
            "median_len": int(lengths.median()),
            "max_len": int(lengths.max()),
        })
    df = pd.DataFrame(rows).sort_values("samples", ascending=False)
    return df.reset_index(drop=True)


def sentence_length_distribution(lid_df: pd.DataFrame) -> plt.Figure:
    """Histogram of sentence lengths per language (clipped at 60 words)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for lang in sorted(lid_df["language"].unique()):
        lens = lid_df.loc[lid_df["language"] == lang, "text"].str.split().str.len()
        lens = lens[lens <= 60]
        ax.hist(lens, bins=40, alpha=0.55, label=lang.capitalize())
    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sentence-length distribution by language")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def src_vs_tgt_lengths(pairs: dict) -> plt.Figure:
    """Scatter + marginals of source vs target lengths (English vs Hindi/Marathi)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"hi": NEXORA_BLUE, "mr": NEXORA_PURPLE}
    for (name, df_pair), ax in zip(pairs.items(), axes):
        src = df_pair["src"].str.split().str.len()
        tgt = df_pair["tgt"].str.split().str.len()
        ax.scatter(src[:3000], tgt[:3000], s=6, alpha=0.35, color=colors[name])
        ax.plot([0, max(src.max(), tgt.max())], [0, max(src.max(), tgt.max())],
                ls="--", c=NEXORA_ACCENT, lw=1.2, label="y = x")
        ax.set_xlabel("Source length (words)")
        ax.set_ylabel("Target length (words)")
        ax.set_title(f"English to {name.upper()} pair lengths")
        ax.legend(fontsize=9)
    fig.suptitle("Source vs target sentence lengths (experimental pairs)", y=1.04)
    fig.tight_layout()
    return fig


def language_distribution(lid_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of sample count per language."""
    counts = lid_df["language"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [NEXORA_PURPLE if c < counts.median() else NEXORA_BLUE for c in counts.values]
    ax.barh([c.capitalize() for c in counts.index], counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(v + v * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of samples")
    ax.set_title("Language distribution in processed dataset")
    fig.tight_layout()
    return fig


def vocabulary_statistics(lid_df: pd.DataFrame) -> plt.Figure:
    """Vocabulary size per language (unique tokens)."""
    rows = []
    for lang in sorted(lid_df["language"].unique()):
        text = " ".join(lid_df.loc[lid_df["language"] == lang, "text"].astype(str))
        tokens = light_tokenize(text)
        rows.append({"language": lang.capitalize(), "vocab": len(set(tokens))})
    df = pd.DataFrame(rows).sort_values("vocab")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(df["language"], df["vocab"], color=NEXORA_TEAL)
    for i, v in enumerate(df["vocab"]):
        ax.text(i, v + v * 0.01, f"{v:,}", ha="center", fontsize=9)
    ax.set_ylabel("Unique tokens")
    ax.set_title("Vocabulary size per language")
    fig.tight_layout()
    return fig


def most_frequent_words(lid_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Bar chart of the most frequent words across all languages."""
    tokens = []
    for lang in lid_df["language"].unique():
        col = lid_df.loc[lid_df["language"] == lang, "text"]
        script = "Devanagari" if lang in ("hindi", "marathi") else detect_script(col.iloc[0])
        t = [t for text in col.astype(str).tolist() for t in light_tokenize(text)]
        tokens.extend(remove_language_stopwords(t, script))
    freq = word_frequencies(tokens, top_n)
    words = [w for w, _ in freq][::-1]
    counts = [c for _, c in freq][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words, counts, color=NEXORA_BLUE)
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} most frequent words (stopwords removed)")
    fig.tight_layout()
    return fig


def missing_duplicate_chart(stats: dict) -> plt.Figure:
    """Missing and duplicated values per language column (raw corpus)."""
    langs = list(stats["duplicates_raw"].keys())
    missing = [stats["missing_raw"][l] for l in langs]
    dup = [stats["duplicates_raw"][l] for l in langs]
    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.2, missing, 0.4, label="Missing", color=NEXORA_ACCENT)
    ax.bar(x + 0.2, dup, 0.4, label="Duplicates", color=NEXORA_PURPLE)
    ax.set_xticks(x, [l.capitalize() for l in langs])
    ax.set_ylabel("Row count")
    ax.set_title("Data quality: missing & duplicate values in raw corpus")
    ax.legend()
    fig.tight_layout()
    return fig


def generate_wordcloud(lid_df: pd.DataFrame, out_path: Path) -> None:
    """Word cloud per language saved as a PNG grid (2 cols)."""
    from wordcloud import WordCloud

    try:
        from wordcloud import STOPWORDS
    except ImportError:  # pragma: no cover
        STOPWORDS = set()

    langs = sorted(lid_df["language"].unique())
    fig, axes = plt.subplots((len(langs) + 1) // 2, 2, figsize=(13, (len(langs) + 1) // 2 * 4.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, lang in zip(axes, langs):
        col = lid_df.loc[lid_df["language"] == lang, "text"].astype(str)
        text = " ".join(col.tolist())
        wc = WordCloud(width=800, height=400, background_color="white",
                       max_words=150, colormap="viridis",
                       stopwords=STOPWORDS | {"_"}).generate(text)
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(lang.capitalize())
    for ax in axes[len(langs):]:
        ax.axis("off")
    fig.suptitle("Word clouds per language — generated from actual data", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vocabulary growth (Seq2Seq corpus)
# ---------------------------------------------------------------------------

def vocab_growth_curve(pairs: dict) -> plt.Figure:
    """Vocabulary size vs number of unique sentences (Zipf-style intuition)."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, df_pair in pairs.items():
        seen = set()
        xs, ys = [], []
        for text in df_pair["src"]:
            seen.add(text)
            xs.append(len(seen))
            ys.append(len(set(t for t in light_tokenize(text))))
        ax.plot(xs[:4000], ys[:4000], label=f"English ({name.upper()})")
    ax.set_xlabel("Unique source sentences seen")
    ax.set_ylabel("Unique tokens")
    ax.set_title("Vocabulary growth on experimental corpus")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aggregate runner (writes all figures + a JSON metrics dump)
# ---------------------------------------------------------------------------

def run_eda(lid_df: pd.DataFrame, pairs: dict, stats: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, path = sentence_length_distribution(lid_df), out_dir / "length_distribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = src_vs_tgt_lengths(pairs); fig.savefig(out_dir / "src_vs_tgt_lengths.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = language_distribution(lid_df); fig.savefig(out_dir / "language_distribution.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = vocabulary_statistics(lid_df); fig.savefig(out_dir / "vocabulary.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = most_frequent_words(lid_df); fig.savefig(out_dir / "frequent_words.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = missing_duplicate_chart(stats); fig.savefig(out_dir / "quality.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    generate_wordcloud(lid_df, out_dir / "wordclouds.png")
    table = compute_dataset_stats(lid_df, stats)
    table.to_csv(out_dir / "dataset_stats.csv", index=False)
    print(f"[eda] wrote figures + stats to {out_dir}")
    return {c: int(v) if _is_int(v) else v for c, v in stats.items()}


def _is_int(v):
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False