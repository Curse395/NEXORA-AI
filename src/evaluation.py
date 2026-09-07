"""
NEXORA — Translation evaluation metrics.

BLEU (corpus-level, smoothed) via NLTK on the experimental model output.
chrF is intentionally not computed on the tiny experimental validation set to
avoid misleading numbers; the metrics file notes that explicitly.
"""

from __future__ import annotations

import io

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def compute_bleu(predictions: list[str], references: list[str], sources: list[str] | None = None) -> dict:
    """
    Corpus BLEU (1–4 gram) with method-3 smoothing. Returns the score plus
    per-gram precisions and brevity penalty, computed from *actual* model
    output vs actual references.
    """
    refs = [[r.split()] for r in references]
    hyps = [p.split() for p in predictions]
    smoother = SmoothingFunction().method3
    bleu = corpus_bleu(refs, hyps, weights=(0.25,) * 4, smoothing_function=smoother)
    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = corpus_bleu(refs, hyps, weights=(0.3333, 0.3333, 0.3333, 0), smoothing_function=smoother)
    return {
        "bleu": float(bleu),
        "bleu1": float(bleu1),
        "bleu2": float(bleu2),
        "bleu3": float(bleu3),
        "n_predictions": len(predictions),
    }


def sample_translation_table(predictions: list[str], references: list[str], sources: list[str], n: int = 8) -> list[dict]:
    """Source / Reference / Prediction rows for the dashboard table."""
    rows = []
    for s, r, p in zip(sources[:n], references[:n], predictions[:n]):
        rows.append({"Source": s, "Reference": r, "Prediction": p})
    return rows