"""
NEXORA — Multilingual text utilities.

Language-aware tokenization, word-frequency analysis and inline
word-cloud helpers for the Indic (Devanagari/Tamil/Telugu/Gujarati)
and Latin scripts covered by the AILA multilingual parallel corpus.
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
_GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_script(text: str) -> str:
    """Return the dominant Unicode block of a piece of text."""
    counts = Counter()
    for ch in text:
        if _DEVANAGARI_RE.match(ch):
            counts["Devanagari"] += 1
        elif _TAMIL_RE.match(ch):
            counts["Tamil"] += 1
        elif _TELUGU_RE.match(ch):
            counts["Telugu"] += 1
        elif _GUJARATI_RE.match(ch):
            counts["Gujarati"] += 1
        elif _BENGALI_RE.match(ch):
            counts["Bengali"] += 1
        elif _LATIN_RE.match(ch):
            counts["Latin"] += 1
    if not counts:
        return "Other"
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Light tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# For Indic scripts, "letters" include combining marks (vowel signs, virama,
# nukta): group runs of base letters + marks + digits into one token.
_INDIC_WORD_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF\u0C00-\u0C7F"
    r"\u0D80-\u0DFF\u0E00-\u0E7F\u0F00-\u0FFF]"
    r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF\u0C00-\u0C7F"
    r"\u0D80-\u0DFF\u0E00-\u0E7F\u0F00-\u0FFF\u200c\u200d]*",
    re.UNICODE,
)

_WORD_RE = re.compile(
    r"[A-Za-z0-9']+|[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF"
    r"\u0C00-\u0C7F\u0D80-\u0DFF\u0E00-\u0E7F\u0F00-\u0FFF]+"
    r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B80-\u0BFF\u0C00-\u0C7F"
    r"\u0D80-\u0DFF\u0E00-\u0E7F\u0F00-\u0FFF\u200c\u200d]*|[^\w\s]",
    re.UNICODE,
)


def light_tokenize(text: str) -> list[str]:
    """
    Script-aware lightweight tokenization.

    - Latin text: word tokens (letters + digits + apostrophes) plus
      punctuation as standalone tokens.
    - Indic text (Devanagari, Bengali, Gujarati, Tamil, Telugu…): tokens are
      *runs of base letters plus combining marks* (matras, virama, nukta),
      which keeps words like ``प्रधानमंत्री`` as ONE token instead of a
      sequence of individual characters.

    This tokenization is used for vocabulary statistics and the experimental
    Seq2Seq model. The production engine uses its own SentencePiece tokenizer.
    """
    text = text.replace("\u200d", "").replace("\u200c", "").strip()
    return [t for t in _WORD_RE.findall(text) if t]


def word_frequencies(tokens: list[str], n: int = 30) -> Counter:
    """Top-n most frequent tokens from a token list."""
    return Counter(tokens).most_common(n)


# ---------------------------------------------------------------------------
# Word cloud helpers (fully offline, no external stop-word lists)
# ---------------------------------------------------------------------------

_LATIN_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has he her his i if in is it its
    of on or she that the their them they this to was were we what when where
    which who will with you your ours our yourselves yourself itself himself
    herself themselves myself ourselves can could did do does doing down each
    few had have having how into just like more most no not out over own same
    so some such than then there these those too up very
    """.split()
)

# Common Indo-Aryan function words (Devanagari script only — these do not
# appear in Tamil/Telugu/Gujarati/Bengali scripts).
_DEVANAGARI_STOPWORDS = frozenset(
    """
    है हैं और का की के को से में पर ने यह ये वह वे इस उस एक दो करने करता करते
    मैं तुम आप हम वो था थी थे था जो कि जब तब भी ही ना नहीं लिए बाद पहले साथ
    तक अपने अपनी अपना हमारे हमारी तुम्हारे तुम्हारी उनके उनकी उसके उसकी इसके
    इसकी इनके इनकी जैसे जैसा जैसी हो होता होते होती हैं हूँ होगा होगी
    """.split()
)


def remove_language_stopwords(tokens: list[str], script: str) -> list[str]:
    """
    Remove common function words for the dominant script. Conservative,
    script-specific lists — we intentionally do NOT aggressively stem or
    lowercase, because doing so damages meaning in morphologically rich
    languages such as Hindi and Marathi.
    """
    if script == "Devanagari":
        stop = _DEVANAGARI_STOPWORDS
    elif script == "Latin":
        stop = _LATIN_STOPWORDS
    else:
        stop = frozenset()
    return [t for t in tokens if t.lower() not in stop]


def build_wordcloud_text(series, header_size: int = 2000) -> str:
    """
    Build a joined text used by the WordCloud generator.
    A random ``header_size``-row sample keeps memory small and the cloud
    representative; the RNG seed keeps results reproducible.
    """
    import random

    import pandas as pd

    if isinstance(series, pd.Series):
        texts = series.dropna().astype(str).tolist()
    else:
        texts = list(series)
    rng = random.Random(42)
    sample = rng.sample(texts, min(header_size, len(texts)))
    return " ".join(sample)