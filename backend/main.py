"""
NEXORA API — FastAPI backend for the premium React frontend.

Exposes every artifact of the project (dataset stats, EDA charts, classical
ML metrics, the experimental Seq2Seq + Attention model and the production
M2M100 translator) as JSON / base64 endpoints, reusing the verified modules
in ``src/``.

Run:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- make the project root importable so `src.*` works ---------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multilingual import light_tokenize  # noqa: E402
from src.translation import LANGUAGES, M2MTranslator  # noqa: E402

MODEL_HOME = ROOT / "models" / "seq2seq"
DATA_HOME = ROOT / "data" / "processed"
OUT_HOME = ROOT / "outputs"

app = FastAPI(title="NEXORA API", version="2.0.0",
              description="Sequence-to-Sequence Networks with Attention-Based Alignment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _img_data_url(path: Path) -> str:
    """Read a PNG and return it as a base64 data URL."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _img_group(files: dict[str, Path]) -> dict[str, str]:
    return {name: _img_data_url(p) for name, p in files.items() if p.exists()}


# ---------------------------------------------------------------------------
# dataset insights (computed lazily once from the parquet)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def dataset_insights() -> dict:
    """Per-language counts + length distribution from the 359k-row LID set."""
    import pandas as pd
    df = pd.read_parquet(DATA_HOME / "lid_dataset.parquet")
    lang_col = "language" if "language" in df.columns else df.columns[1]
    counts = df[lang_col].value_counts().sort_index()
    langs = {k: v["name"] for k, v in LANGUAGES.items()}
    dist = [
        {"code": c, "name": langs.get(c, c), "count": int(n)}
        for c, n in counts.items()
    ]
    # length distribution (sampled for speed)
    text_col = "text" if "text" in df.columns else df.columns[0]
    lens = (df[text_col].str.len() // 20 * 20).clip(upper=600).value_counts().sort_index()
    lengths = [
        {"bucket": int(b), "count": int(n)}
        for b, n in lens.items()
    ]
    return {"languages": dist, "lengths": lengths}


# ---------------------------------------------------------------------------
# cached machine-translation engine + experimental model
# ---------------------------------------------------------------------------

_translator: M2MTranslator | None = None


def get_translator() -> M2MTranslator:
    global _translator
    if _translator is None:
        _translator = M2MTranslator()
    return _translator


@functools.lru_cache(maxsize=1)
def seq2seq_bundle() -> dict:
    """Load vocab, weights and precomputed attention samples once."""
    import numpy as np
    import torch

    from src.seq2seq import PAD_IDX, Seq2SeqAttention, Vocab

    meta = _read_json(MODEL_HOME / "metrics.json")
    vs = _read_json(MODEL_HOME / "vocab_src.json")
    vt = _read_json(MODEL_HOME / "vocab_tgt.json")
    vocab_src = Vocab(min_freq=2, max_size=10_000)
    vocab_src.stoi, vocab_src.itos = vs["stoi"], list(vs["itos"])
    vocab_tgt = Vocab(min_freq=2, max_size=10_000)
    vocab_tgt.stoi, vocab_tgt.itos = vt["stoi"], list(vt["itos"])
    model = Seq2SeqAttention(meta["vocab_src_size"], meta["vocab_tgt_size"],
                             meta["embed_dim"], meta["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_HOME / "seq2seq_attention.pt",
                                     map_location="cpu"))
    model.eval()
    attn = np.load(MODEL_HOME / "attention_sample.npy")
    samples = json.loads((MODEL_HOME / "sample_translations.json").read_text(encoding="utf-8"))
    return {"model": model, "vocab_src": vocab_src, "vocab_tgt": vocab_tgt,
            "meta": meta, "attn": attn, "samples": samples}


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    src: str = "en"
    tgt: str = "hi"


class AttentionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "nexora-api", "version": "2.0.0"}


@app.get("/api/languages")
async def languages():
    return [{"code": c, "name": v["name"], "flag": v["flag"]} for c, v in LANGUAGES.items()]


@app.get("/api/overview")
async def overview():
    """Everything the landing dashboard needs in one call."""
    stats = _read_json(DATA_HOME / "dataset_stats.json")
    classical = _read_json(OUT_HOME / "metrics" / "classical_metrics.json")
    seq = _read_json(MODEL_HOME / "metrics.json")
    return {"stats": stats, "classical": classical, "seq2seq": seq,
            "languages": await languages()}


@app.get("/api/eda")
async def eda_charts():
    plots = OUT_HOME / "plots"
    charts = {
        "language_distribution": _img_data_url(plots / "language_distribution.png"),
        "length_distribution": _img_data_url(plots / "length_distribution.png"),
        "src_vs_tgt_lengths": _img_data_url(plots / "src_vs_tgt_lengths.png"),
        "vocabulary": _img_data_url(plots / "vocabulary.png"),
        "frequent_words": _img_data_url(plots / "frequent_words.png"),
        "wordclouds": _img_data_url(plots / "wordclouds.png"),
        "quality": _img_data_url(plots / "quality.png"),
    }
    charts = {k: v for k, v in charts.items() if v}
    return {
        "charts": charts,
        "insights": dataset_insights(),
        "stats": _read_json(DATA_HOME / "dataset_stats.json"),
    }


@app.get("/api/dataset-insights")
async def insights():
    return dataset_insights()


@app.get("/api/classical")
async def classical():
    metrics_dir = OUT_HOME / "metrics"
    payload = _read_json(metrics_dir / "classical_metrics.json")
    payload["charts"] = _img_group({
        "naive_bayes": metrics_dir / "confusion_naive_bayes.png",
        "logistic_regression": metrics_dir / "confusion_logistic_regression.png",
        "svm": metrics_dir / "confusion_svm.png",
        "comparison": metrics_dir / "model_comparison.png",
        "roc": metrics_dir / "roc_curves.png",
    })
    return payload


@app.get("/api/seq2seq")
async def seq2seq():
    meta = _read_json(MODEL_HOME / "metrics.json")
    samples = json.loads((MODEL_HOME / "sample_translations.json").read_text(encoding="utf-8"))
    return {"metrics": meta, "samples": samples}


@app.get("/api/attention/samples")
async def attention_samples():
    """Precomputed alignment samples — no model load required."""
    bundle = seq2seq_bundle()
    attn, samples = bundle["attn"], bundle["samples"]
    out = []
    for i in range(min(attn.shape[0], len(samples))):
        src_tokens = light_tokenize(samples[i]["source"])
        tgt_tokens = samples[i]["prediction"].split()
        w = attn[i]  # (tgt_len, src_len)
        n_tgt, n_src = w.shape
        w = w[: max(len(tgt_tokens), 1), : max(len(src_tokens), 1)]
        out.append({
            "source": samples[i]["source"],
            "reference": samples[i]["reference"],
            "prediction": samples[i]["prediction"],
            "src_tokens": src_tokens[: w.shape[1]],
            "tgt_tokens": tgt_tokens[: w.shape[0]],
            "weights": [[round(float(v), 4) for v in row] for row in w],
        })
    return {"samples": out}


@app.post("/api/attention")
async def attention_live(req: AttentionRequest):
    """Live attention from the trained experimental model on the user's text."""
    bundle = seq2seq_bundle()
    model, vocab_src, vocab_tgt = bundle["model"], bundle["vocab_src"], bundle["vocab_tgt"]
    max_len = 20
    loop = asyncio.get_event_loop()
    preds, weights = await loop.run_in_executor(
        None, lambda: model.translate([req.text], vocab_src, vocab_tgt, max_len=max_len)
    )
    pred = preds[0]
    w = weights[0]  # (tgt_len, src_len)
    src_tokens = light_tokenize(req.text)
    tgt_tokens = pred.split()
    n_tgt, n_src = w.shape
    w = w[: max(len(tgt_tokens), 1), : max(len(src_tokens), 1)]
    return {
        "source": req.text,
        "prediction": pred,
        "src_tokens": src_tokens[: w.shape[1]],
        "tgt_tokens": tgt_tokens[: w.shape[0]],
        "weights": [[round(float(v), 4) for v in row] for row in w],
    }


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    if req.src not in LANGUAGES or req.tgt not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language code.")
    src_m2m, tgt_m2m = LANGUAGES[req.src]["m2m"], LANGUAGES[req.tgt]["m2m"]
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: get_translator().translate(req.text, src_m2m, tgt_m2m)
        )
    except Exception as exc:  # model download / offline
        raise HTTPException(status_code=503, detail=f"Translation engine unavailable: {exc}")
    return {
        "source": result.source,
        "translated": result.translated,
        "source_lang": req.src,
        "target_lang": req.tgt,
        "engine": result.engine,
    }