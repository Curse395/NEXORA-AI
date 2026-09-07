"""
NEXORA — Production translation engine.

Uses the pretrained ``facebook/m2m100_418M`` multilingual model from Meta AI
(olive oil trained on 7.5B parallel sentences across 100+ languages) for
live, high-quality translation.

IMPORTANT (academic honesty):
- This is a PRETRAINED model — NEXORA does not claim to have trained it.
- Our own training is the *experimental* Seq2Seq + Attention model in
  ``src/seq2seq.py``, used for the attention-alignment demos.
- M2M100 supports 100 languages; files are cached after first download and
  the app degrades to English fallback messaging if offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Supported languages (subset of M2M100's 100 languages) -------------------
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, dict] = {
    "en": {"name": "English", "flag": "🇬🇧", "m2m": "en"},
    "hi": {"name": "Hindi", "flag": "🇮🇳", "m2m": "hi"},
    "mr": {"name": "Marathi", "flag": "🇮🇳", "m2m": "mr"},
    "bn": {"name": "Bengali", "flag": "🇧🇩", "m2m": "bn"},
    "gu": {"name": "Gujarati", "flag": "🇮🇳", "m2m": "gu"},
    "ta": {"name": "Tamil", "flag": "🇮🇳", "m2m": "ta"},
    "te": {"name": "Telugu", "flag": "🇮🇳", "m2m": "te"},
    "fr": {"name": "French", "flag": "🇫🇷", "m2m": "fr"},
    "es": {"name": "Spanish", "flag": "🇪🇸", "m2m": "es"},
    "de": {"name": "German", "flag": "🇩🇪", "m2m": "de"},
}
DEFAULT_LANGS = ("en", "hi", "mr", "bn", "gu", "ta", "te", "fr", "es", "de")
CACHE_DIR = Path("models/m2m100")

MODEL_NAME = "facebook/m2m100_418M"


@dataclass
class TranslationResult:
    source: str
    target: str
    translated: str
    source_lang: str
    target_lang: str
    engine: str = "facebook/m2m100_418M (pretrained)"
    info: dict = field(default_factory=dict)


class M2MTranslator:
    """Lazy-loading wrapper around the M2M100 model + tokenizer."""

    # M2M100_418M needs roughly 2-4 GB of RAM once loaded (plus the ~1.9 GB
    # download). Streamlit Community Cloud free instances only guarantee 1 GB,
    # so we refuse to load there and fail FAST with a friendly message instead
    # of OOM-ing the whole app.
    MIN_RAM_MB = 1800

    def __init__(self, use_cache: bool = True):
        self.model = None
        self.tokenizer = None
        self._dir = CACHE_DIR if use_cache else None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        import psutil
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        # ---- memory guard --------------------------------------------------
        try:
            avail_mb = psutil.virtual_memory().available / (1024**2)
        except Exception:
            avail_mb = float("inf")
        if avail_mb < self.MIN_RAM_MB:
            raise MemoryError(
                f"Not enough free memory to load the M2M100 model "
                f"({avail_mb:.0f} MB free, need ≥ {self.MIN_RAM_MB} MB). "
                f"The Live Translation engine is disabled on this plan — "
                f"everything else in NEXORA works fine."
            )

        kwargs = {}
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            kwargs = {"cache_dir": str(self._dir)}
        self.tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME, **kwargs)
        self.model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self._device = device

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def translate(self, text: str, source_lang: str, target_lang: str, max_length: int = 128) -> TranslationResult:
        import torch

        self._ensure_loaded()
        self.tokenizer.src_lang = source_lang
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                max_length=max_length,
                num_beams=1,
            )
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return TranslationResult(
            source=text,
            target=out,
            translated=out,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def detect_supported_languages(self) -> list[str]:
        """List of M2M100 language names the tokenizer actually supports."""
        self._ensure_loaded()
        return sorted(self.tokenizer.lang_code_to_id.keys())

    def beam_search_translate(self, text, source_lang, target_lang, num_beams=5):
        """Beam search variant used by the 'Advanced' demo in the app."""
        import torch

        self._ensure_loaded()
        self.tokenizer.src_lang = source_lang
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                max_length=128,
                num_beams=num_beams,
            )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


_translator: M2MTranslator | None = None


def get_translator() -> M2MTranslator:
    """Module-level singleton so the model loads once per session."""
    global _translator
    if _translator is None:
        _translator = M2MTranslator()
    return _translator