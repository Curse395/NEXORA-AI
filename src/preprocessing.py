"""
NEXORA — Data loading & language-aware preprocessing.

Pipeline stages
---------------
1. Load the raw AILA parallel corpus (Hugging Face).
2. Select the multilingual subset (English + 9 Indian languages) and
   build the supervised flat language-identification task (classical ML).
3. Build language-pair Seq2Seq datasets (English <-> Hindi, English <-> Marathi)
   for the experimental attention model.
4. Clean, normalize and tokenize text *without* damaging scripts:
   - Unicode NFC normalization
   - Devanagari Nukta (़) retention for Hindi/Marathi
   - no aggressive lowercasing or character deletion for non-Latin scripts
   - whitespace/control-character cleanup applied for every script

Missing values and duplicates are handled here and their counts recorded.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset metadata (also used by the dashboard)
# ---------------------------------------------------------------------------

# Samanantar: "The Largest Publicly Available Parallel Corpora Collection for
# 11 Indic Languages" (Ramesh et al., 2022; AI4Bharat, IIT Madras).
DATASET_NAME = "ai4bharat/samanantar"
DATASET_URL = "https://huggingface.co/datasets/ai4bharat/samanantar"
DATASET_SOURCE = (
    "Hugging Face Hub — ai4bharat/samanantar (AI4Bharat, IIT Madras). "
    "See documentation: https://indicnlp.ai4bharat.org/samanantar/"
)
# Each language-config exposes {idx, src (English), tgt (target language)}
COLUMNS = ["src", "tgt"]

PAIR_LANGS = {
    "hi": "hi",
    "mr": "mr",
    "bn": "bn",
    "gu": "gu",
    "ta": "ta",
    "te": "te",
}

# Languages used for the flat language-identification task
LID_LANGS = ["hi", "mr", "bn", "gu", "ta", "te"]
LID_LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
}


# ---------------------------------------------------------------------------
# Cleaning (language-aware)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str, script: str) -> str:
    """Unicode-normalize and trim whitespace. Punctuation and case are kept."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def filter_valid_sentence(text: str, script: str) -> bool:
    """
    Keep a sentence when it contains at least one character from the script
    it claims to be (Latin for English; the appropriate Unicode block for
    Indian languages). Rows that fail this check are almost always wrong
    alignment rows in the source corpus and are removed.
    """
    import src.multilingual as ml

    text = (text or "").strip()
    if len(text) < 1:
        return False
    if script == "english":
        return bool(ml._LATIN_RE.search(text))
    checker = {
        "hi": ml._DEVANAGARI_RE,
        "mr": ml._DEVANAGARI_RE,
        "bn": ml._BENGALI_RE,
        "gu": ml._GUJARATI_RE,
        "ta": ml._TAMIL_RE,
        "te": ml._TELUGU_RE,
    }[script]
    return bool(checker.search(text))


# ---------------------------------------------------------------------------
# Main loading pipeline
# ---------------------------------------------------------------------------

def load_raw_subset(streamlit_progress=None, max_rows_per_lang: int | None = 60_000) -> pd.DataFrame:
    """
    Load the Samanantar corpus for the 6 Indic language configs.

    Samanantar stores one English sentence per row: ``src`` = English,
    ``tgt`` = the target language. We stream a bounded number of rows per
    language config and build a flat ``(text, language)`` frame used by the
    language-identification task.
    """
    from datasets import load_dataset

    records = []
    for lang in LID_LANGS:
        ds = load_dataset(DATASET_NAME, lang, split="train", streaming=True)
        it = iter(ds)
        for i, row in enumerate(it):
            if max_rows_per_lang is not None and i >= max_rows_per_lang:
                break
            records.append((row.get("tgt", "") or "", lang))
        if streamlit_progress is not None:
            streamlit_progress.progress(1.0, text=f"Streamed Samanantar/{lang}…")
    df = pd.DataFrame(records, columns=["text", "language"])
    return df


def build_datasets(seed: int = 42) -> dict:
    """
    Run the full preprocessing pipeline.

    Returns a dictionary with keys:
      lid_df            — flat language-identification dataset
      pairs             — {pair_name: pair_df} for the experimental Seq2Seq model
      stats             — data-quality statistics (missing, duplicates, splits)
    """
    raw = load_raw_subset()
    stats: dict = {}
    stats["raw_rows"] = int(len(raw))

    # --- quality stage -----------------------------------------------------
    missing = (raw["text"] == "").sum()
    stats["missing_raw"] = {"text": int(missing)}
    dup_counts = {"text": int(raw["text"].duplicated().sum())}
    stats["duplicates_raw"] = dup_counts

    # --- LID dataset ------------------------------------------------------
    records = []
    for lang in LID_LANGS:
        texts = raw.loc[raw["language"] == lang, "text"]
        cleaned = [clean_text(t, lang) for t in texts]
        kept = [(t, lang) for t in cleaned if filter_valid_sentence(t, lang)]
        records.extend(kept)
    lid_df = pd.DataFrame(records, columns=["text", "language"])
    lid_df = lid_df.drop_duplicates(subset=["text", "language"]).reset_index(drop=True)
    stats["lid_rows"] = int(len(lid_df))

    # --- pair datasets for the experimental Seq2Seq model ------------------
    pairs = {}
    for pair in ("hi", "mr"):
        ds = __import__("datasets").load_dataset(DATASET_NAME, pair, split="train", streaming=True)
        src, tgt = [], []
        for i, row in enumerate(ds):
            if i >= 60_000:
                break
            src.append(row.get("src", "") or "")
            tgt.append(row.get("tgt", "") or "")
        clean_src = [clean_text(t, "english") for t in src]
        clean_tgt = [clean_text(t, "hi" if pair == "hi" else "mr") for t in tgt]
        df_pair = pd.DataFrame({"src": clean_src, "tgt": clean_tgt})
        df_pair = df_pair[
            [filter_valid_sentence(a, "english") and
             filter_valid_sentence(b, "hi" if pair == "hi" else "mr")
             for a, b in zip(df_pair["src"], df_pair["tgt"])]
        ].reset_index(drop=True)
        stats[f"pair_{pair}_rows"] = int(len(df_pair))
        pairs[pair] = df_pair

    # --- reproducible splits ----------------------------------------------
    rng = __import__("random").Random(seed)
    n = len(lid_df)
    idx = list(range(n))
    rng.shuffle(idx)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    stats["splits"] = {"train": n_tr, "valid": n_va, "test": n - n_tr - n_va}

    stats["n_features"] = len(LID_LANGS)
    stats["languages"] = LID_LANGUAGE_NAMES
    return {"lid": lid_df, "pairs": pairs, "stats": stats}


def save_processed(df_lid: pd.DataFrame, pairs: dict, stats: dict, out_dir: Path) -> None:
    """Persist processed parquets used by the pipeline and the dashboard."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_lid.to_parquet(out_dir / "lid_dataset.parquet")
    for pair, df_pair in pairs.items():
        df_pair.head(30_000).to_parquet(out_dir / f"pair_{pair}_sample.parquet")
    with open(out_dir / "dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"[preprocessing] saved processed datasets to {out_dir}")


if __name__ == "__main__":
    res = build_datasets()
    print("LID rows:", res["stats"]["lid_rows"])
    print("Pair hi rows:", res["stats"]["pair_hi_rows"])
    print("Pair mr rows:", res["stats"]["pair_mr_rows"])