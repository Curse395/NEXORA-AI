"""
NEXORA — Build pipeline.

Runs the full machine-learning pipeline end-to-end and writes reproducible
artifacts under ``data/processed``, ``outputs/`` and ``models/``:

  1. preprocessing  -> processed datasets
  2. EDA            -> figures + stats (from actual data)
  3. features       -> CountVectorizer / TF-IDF matrices
  4. classical ML   -> NB / LR / SVM metrics + plots
  5. seq2seq        -> experimental attention model + BLEU + attention maps

Usage:
    python build_pipeline.py --steps preprocess,eda,classical,seq2seq
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import build_datasets, save_processed
from src import eda as eda_mod
from src.features import build_features
from src import classical_models as cm
from src.seq2seq import train_model

PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "outputs" / "plots"
METRICS = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "models" / "seq2seq"


def run_preprocess(quick: bool = False) -> dict:
    print("\n== [1/5] preprocessing ==", flush=True)
    res = build_datasets()
    save_processed(res["lid"], res["pairs"], res["stats"], PROCESSED)
    stats = res["stats"]
    print(f"  LID rows: {stats['lid_rows']:,}")
    print(f"  pair hi:  {stats['pair_hi_rows']:,}")
    print(f"  pair mr:  {stats['pair_mr_rows']:,}")
    return res


def run_eda(lid_df, pairs, stats) -> None:
    print("\n== [2/5] EDA ==", flush=True)
    eda_mod.run_eda(lid_df, pairs, stats, PLOTS)


def run_features(lid_df) -> dict:
    print("\n== [3/5] feature engineering ==", flush=True)
    feats = build_features(lid_df, save_dir=PROCESSED)
    print("  feature spaces ready:", list(feats["fitted"].keys()))
    return feats


def run_classical(feats) -> None:
    print("\n== [4/5] classical models ==", flush=True)
    payload = cm.train_all(feats, save_dir=METRICS)
    cm.plot_confusion_matrices(payload, METRICS)
    roc_path = cm.plot_roc_curves(payload, feats, METRICS)
    cm.plot_model_comparison(payload, METRICS)
    print("  confusion + comparison written", flush=True)
    if roc_path is None:
        print("  ROC skipped (no predict_proba on LinearSVC singleton)", flush=True)


def run_seq2seq(pairs, quick: bool = False) -> None:
    print("\n== [5/5] seq2seq ==", flush=True)
    pair_df = pairs["hi"]
    sample = 25_000 if quick else 30_000
    epochs = 2 if quick else 3
    meta, model, vocab_src, vocab_tgt = train_model(
        pair_df, MODEL_DIR, sample=sample, epochs=epochs,
        max_len=20, batch_size=64,
    )
    print(f"  BLEU: {meta['bleu']['bleu']:.4f}", flush=True)

    # Render real attention heatmaps for the first sample translations
    try:
        import json as _json
        import numpy as _np

        samples = _json.loads((MODEL_DIR / "sample_translations.json").read_text(encoding="utf-8"))
        attn = _np.load(MODEL_DIR / "attention_sample.npy")
        from src.attention import save_sample_attention
        paths = save_sample_attention(
            attn, [s["source"] for s in samples],
            [s["reference"] for s in samples],
            [s["prediction"] for s in samples],
            vocab_src, vocab_tgt, ROOT / "outputs" / "attention", n=4,
        )
        print(f"  attention heatmaps: {len(paths)} saved", flush=True)
    except Exception as e:
        print(f"  attention heatmap render skipped: {e}", flush=True)
    print("[pipeline] done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="comma list or 'all': preprocess,eda,classical,seq2seq")
    ap.add_argument("--quick", action="store_true",
                    help="smaller corpus + shorter training for CI")
    args = ap.parse_args()

    steps = ["preprocess", "eda", "classical", "seq2seq"] if args.steps == "all" \
        else [s.strip() for s in args.steps.split(",")]

    t0 = time.time()
    res = None
    if "preprocess" in steps:
        res = run_preprocess(args.quick)
    if "eda" in steps:
        if res is None:
            res = _load_cached()
        run_eda(res["lid"], res["pairs"], res["stats"])
    feats = None
    if "classical" in steps:
        if res is None:
            res = _load_cached()
        feats = run_features(res["lid"])
        run_classical(feats)
    if "seq2seq" in steps:
        if res is None:
            res = _load_cached()
        run_seq2seq(res["pairs"], args.quick)
    print(f"\nWall time: {time.time() - t0:.1f}s")


def _load_cached() -> dict:
    import pandas as pd

    lid = pd.read_parquet(PROCESSED / "lid_dataset.parquet")
    stats = json.loads((PROCESSED / "dataset_stats.json").read_text())
    pairs = {}
    for name in ("hi", "mr"):
        p = PROCESSED / f"pair_{name}_sample.parquet"
        if p.exists():
            pairs[name] = pd.read_parquet(p)
    return {"lid": lid, "pairs": pairs, "stats": stats}


if __name__ == "__main__":
    main()