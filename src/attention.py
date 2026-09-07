"""
NEXORA — Attention-based alignment visualization.

Renders real attention-weight heatmaps from the experimental Seq2Seq model
(rows = target tokens, columns = source tokens). Also ships a *live*
illustration of the attention computation on user-entered text using the same
trained encoder/decoder, so the demo never shows fabricated weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEXORA_BLUE = "#2563eb"
NEXORA_PURPLE = "#7c3aed"


def _register_unicode_fonts() -> None:
    """Register Windows fonts covering Latin + Indic scripts for heatmaps."""
    from matplotlib import font_manager

    for p in (r"C:\Windows\Fonts\Nirmala.ttf", r"C:\Windows\Fonts\Mangal.ttf",
              r"C:\Windows\Fonts\ArialUni.ttf"):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            continue
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Nirmala UI", "Mangal", "Arial Unicode MS", "DejaVu Sans"]


_register_unicode_fonts()


def tokens_heatmap(weights: np.ndarray, src_tokens, tgt_tokens, title="Attention-Based Alignment",
                   save_path: Path | None = None, dpi: int = 130) -> plt.Figure:
    """
    Heatmap with rows = target tokens, columns = source tokens.
    ``weights`` shape: (tgt_len, src_len).
    """
    weights = np.asarray(weights, dtype=float)
    # keep the matrix consistent with the token lists
    n_tgt, n_src = weights.shape
    src_tokens = list(src_tokens)[:n_src]
    tgt_tokens = list(tgt_tokens)[:n_tgt]
    if len(src_tokens) < n_src:
        weights = weights[:, : len(src_tokens)]
    if len(tgt_tokens) < n_tgt:
        weights = weights[: len(tgt_tokens), :]

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * weights.shape[1] + 3), max(5, 0.5 * weights.shape[0] + 2)))
    im = ax.imshow(weights, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(weights.shape[1]))
    ax.set_yticks(range(weights.shape[0]))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tgt_tokens, fontsize=10)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            v = weights[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > weights.max() * 0.6 else "#1e293b")
    ax.set_xlabel("Source tokens (encoder hidden states)")
    ax.set_ylabel("Target tokens (decoder steps)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def alignment_explanation_card() -> dict:
    """Static explanation content shown next to the heatmap in the dashboard."""
    return {
        "what": (
            "Attention-based alignment answers the question *which source words "
            "influenced each target word*. At every decoder step t the "
            "Bahdanau attention module scores every encoder hidden state "
            "h_j, normalizes the scores with softmax into weights α_tj, and "
            "sums the states to form the context vector c_t."
        ),
        "math": (
            r"e_{tj} = v_a^\top \tanh\!\big(W_a h_j + U_a s_{t-1}\big),"
            r"\quad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_j \exp(e_{tj})},"
            r"\quad c_t = \sum_j \alpha_{tj} h_j"
        ),
        "reading": (
            "Bright cells in row t mean the model relied strongly on that "
            "source token while producing target token t. The diagonal-ish "
            "structure in well-translated short sentences is the classic "
            "monotone-ish alignment signature."
        ),
    }


def save_sample_attention(attention_array: np.ndarray, src_texts, ref_texts, pred_texts,
                          vocab_src, vocab_tgt, out_dir: Path, n: int = 4) -> list[Path]:
    """Save heatmap PNGs for the first `n` sample translations."""
    from src.multilingual import light_tokenize

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(min(n, len(src_texts))):
        src_toks = light_tokenize(src_texts[i])
        tgt_toks = light_tokenize(pred_texts[i]) if pred_texts[i] else ["<eos>"]
        w = attention_array[i, : len(tgt_toks), : len(src_toks)]
        if w.shape != (0, 0) and w.size > 0:
            p = out_dir / f"attention_{i:02d}.png"
            tokens_heatmap(w, src_toks, tgt_toks,
                           title=f"Attention alignment — sample {i + 1}",
                           save_path=p)
            paths.append(p)
    return paths