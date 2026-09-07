"""
NEXORA — Streamlit dashboard (premium edition).

Brand        NEXORA  ·  AI That Connects Meaning
Topic        Machine Translation using Sequence-to-Sequence Networks
             with Attention-Based Alignment

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import COLUMNS, DATASET_NAME, DATASET_SOURCE, DATASET_URL
from src.translation import LANGUAGES, MODEL_NAME

# Register Unicode fonts (Latin + Indic scripts) for matplotlib charts
from src.attention import _register_unicode_fonts as _reg_fonts

_reg_fonts()

st.set_page_config(
    page_title="NEXORA — Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

_NEXORA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --nx-blue:#3b82f6;
  --nx-violet:#8b5cf6;
  --nx-cyan:#22d3ee;
  --nx-text:#e8ecf4;
  --nx-muted:#8f98ad;
  --nx-card:rgba(255,255,255,.045);
  --nx-border:rgba(255,255,255,.09);
  --nx-glow:0 14px 44px rgba(2,6,17,.5);
}
html, body, .stApp{
  background:
    radial-gradient(1100px 700px at 12% -8%, rgba(59,130,246,.16), transparent 60%),
    radial-gradient(900px 620px at 108% 6%, rgba(139,92,246,.16), transparent 55%),
    radial-gradient(1000px 800px at 50% 125%, rgba(34,211,238,.09), transparent 60%),
    linear-gradient(180deg,#060a13 0%,#0a1120 55%,#0d1526 100%);
  color:var(--nx-text);
  font-family:'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.stApp{ animation:nx-fade-in .55s ease both; }
header[data-testid="stHeader"]{ background:transparent; backdrop-filter:blur(8px); }
#MainMenu, footer{ visibility:hidden; }
[data-testid="stMainBlockContainer"]{ padding-top:2.2rem; max-width:1200px; }

/* ---------------- keyframes ---------------- */
@keyframes nx-fade-in{ from{opacity:0} to{opacity:1} }
@keyframes nx-rise{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
@keyframes nx-rise-sm{ from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none} }
@keyframes nx-grad-move{ 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes nx-orb{ 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(26px,-22px) scale(1.08)} }

/* ---------------- sidebar ---------------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(10,17,32,.94), rgba(8,13,25,.94));
  border-right:1px solid rgba(255,255,255,.07);
  backdrop-filter:blur(18px);
}
[data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,.08); }
[data-testid="stSidebar"] [role="radiogroup"] label{
  display:flex; align-items:center; gap:.75rem;
  padding:.55rem .95rem; margin:.1rem 0; border-radius:12px;
  border:1px solid transparent; cursor:pointer;
  transition:background .22s, transform .22s, border-color .22s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:rgba(255,255,255,.06); transform:translateX(3px);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{ display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(135deg, rgba(59,130,246,.16), rgba(139,92,246,.16));
  border-color:rgba(96,165,250,.38);
  box-shadow:0 4px 20px rgba(59,130,246,.15);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{ color:#fff; font-weight:700; }
[data-testid="stSidebar"] [role="radiogroup"] p{ color:#a7b0c3; font-weight:500; }

/* ---------------- inputs ---------------- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea{
  background:rgba(255,255,255,.055);
  border:1px solid rgba(255,255,255,.1);
  border-radius:14px;
  color:var(--nx-text);
  font-family:'Inter', sans-serif;
  transition:border-color .25s, box-shadow .25s, background .25s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
  border-color:rgba(96,165,250,.6);
  box-shadow:0 0 0 3px rgba(59,130,246,.18);
  background:rgba(255,255,255,.08);
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder{ color:#5b6478; }
[data-testid="stSelectbox"] > div > div{
  background:rgba(255,255,255,.055);
  border:1px solid rgba(255,255,255,.1);
  border-radius:14px; color:var(--nx-text);
  transition:border-color .25s, box-shadow .25s;
}
[data-testid="stSelectbox"] > div > div:focus-within{
  border-color:rgba(96,165,250,.6); box-shadow:0 0 0 3px rgba(59,130,246,.18);
}
[data-testid="stSelectbox"] [role="combobox"]{ color:var(--nx-text) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li{
  background:#0d1526 !important; color:var(--nx-text) !important;
}

/* ---------------- buttons ---------------- */
.stButton > button{
  background:linear-gradient(135deg,#3b82f6,#8b5cf6);
  color:#fff; border:none; border-radius:13px;
  font-weight:700; font-size:.95rem; letter-spacing:.01em;
  padding:.62rem 1.5rem;
  box-shadow:0 6px 24px rgba(59,130,246,.30);
  transition:transform .25s cubic-bezier(.4,0,.2,1), box-shadow .25s, filter .25s;
}
.stButton > button:hover{
  transform:translateY(-2px); filter:brightness(1.08);
  box-shadow:0 10px 34px rgba(139,92,246,.42); color:#fff;
}
.stButton > button:active{ transform:translateY(0); }

/* ---------------- alerts ---------------- */
[data-testid="stAlert"]{
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);
  border-radius:14px;
  backdrop-filter:blur(10px);
  color:var(--nx-text);
}

/* ---------------- tabs ---------------- */
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"]{
  background:transparent; border-radius:10px 10px 0 0;
  color:#9aa3b7; padding:.5rem 1.05rem; font-weight:600;
  transition:color .25s, background .25s;
}
.stTabs [data-baseweb="tab"]:hover{ color:#fff; background:rgba(255,255,255,.06); }
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color:#fff; background:rgba(59,130,246,.12);
  box-shadow:inset 0 -2px 0 #3b82f6;
}
.stTabs [data-baseweb="tab-highlight"]{ background:#3b82f6; }
.stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------------- metrics ---------------- */
[data-testid="stMetric"]{
  background:var(--nx-card);
  border:1px solid var(--nx-border);
  border-radius:16px;
  padding:1rem 1.25rem;
  backdrop-filter:blur(12px);
  transition:transform .3s, border-color .3s, box-shadow .3s;
  animation:nx-rise-sm .6s cubic-bezier(.4,0,.2,1) both;
}
[data-testid="stMetric"]:hover{
  transform:translateY(-3px);
  border-color:rgba(96,165,250,.4);
  box-shadow:var(--nx-glow);
}
[data-testid="stMetricLabel"]{ color:var(--nx-muted); font-weight:600; letter-spacing:.03em; }
[data-testid="stMetricValue"]{ color:#fff; font-weight:800; font-size:1.75rem !important; }
[data-testid="stMetricDeltaIcon"]{ display:none; }
[data-testid="stMetricDelta"]{ color:#34d399; }

/* ---------------- dataframe / json / caption ---------------- */
[data-testid="stDataFrame"]{
  border-radius:16px; overflow:hidden; border:1px solid var(--nx-border);
  box-shadow:var(--nx-glow);
}
[data-testid="stJson"]{
  background:rgba(255,255,255,.04); border:1px solid var(--nx-border);
  border-radius:16px;
}
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] a{ color:var(--nx-muted); }

/* ---------------- expander ---------------- */
[data-testid="stExpander"]{
  background:var(--nx-card); border:1px solid var(--nx-border);
  border-radius:16px; backdrop-filter:blur(12px);
  transition:border-color .3s;
}
[data-testid="stExpander"] summary{ font-weight:600; color:#dbe2ee; }
[data-testid="stExpander"]:hover{ border-color:rgba(96,165,250,.35); }

/* ---------------- images ---------------- */
[data-testid="stImage"] img{
  border-radius:16px; border:1px solid var(--nx-border); box-shadow:var(--nx-glow);
}
[data-testid="stImage"] figcaption{ color:var(--nx-muted); }

/* ---------------- spinner ---------------- */
[data-testid="stStatusWidget"] svg{ color:#3b82f6; }

/* ---------------- product classes ---------------- */
.nx-rise{ animation:nx-rise .7s cubic-bezier(.4,0,.2,1) both; }
.nx-grad-text{
  background:linear-gradient(120deg,#60a5fa,#a78bfa,#22d3ee,#60a5fa);
  background-size:280% 280%;
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:transparent;
  animation:nx-grad-move 9s ease infinite;
}
.nx-card{
  background:var(--nx-card);
  border:1px solid var(--nx-border);
  border-radius:18px;
  padding:1.15rem 1.35rem;
  backdrop-filter:blur(14px);
  transition:transform .3s cubic-bezier(.4,0,.2,1), border-color .3s, box-shadow .3s;
  animation:nx-rise .7s cubic-bezier(.4,0,.2,1) both;
}
.nx-card:hover{
  transform:translateY(-4px);
  border-color:rgba(96,165,250,.38);
  box-shadow:var(--nx-glow), 0 0 0 1px rgba(96,165,250,.12);
}
.nx-kicker{
  display:inline-flex; align-items:center; gap:.45rem;
  font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;
  color:#93b4fb;
  background:rgba(59,130,246,.12);
  border:1px solid rgba(59,130,246,.28);
  padding:.32rem .95rem; border-radius:999px;
  animation:nx-rise-sm .6s ease both;
}
.nx-mono{ font-family:'JetBrains Mono', monospace; }
</style>
"""


def inject_css() -> None:
    """Inject the premium design system (call once, right after config)."""
    st.markdown(_NEXORA_CSS, unsafe_allow_html=True)


def kicker(text: str) -> None:
    st.markdown(f'<div class="nx-kicker">{text}</div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None, kicker_text: str | None = None) -> None:
    k = f'<div class="nx-kicker">{kicker_text}</div>' if kicker_text else ""
    sub = (
        f'<div style="color:#8f98ad; font-size:.95rem; margin-top:.30rem;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="nx-rise" style="margin:.2rem 0 1.1rem 0;">
            {k}
            <div style="font-size:2.05rem; font-weight:800; letter-spacing:-.01em;
                        color:#fff; line-height:1.15; margin-top:.5rem;">
                {title}
            </div>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def nex_card(title: str, body: str, icon: str = "✦", delay: float = 0.0) -> None:
    st.markdown(
        f'<div class="nx-card" style="animation-delay:{delay:.2f}s;">'
        f'<div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;">'
        f'<span style="font-size:1.15rem;">{icon}</span>'
        f'<span style="font-weight:700; color:#fff; font-size:1.03rem;">{title}</span></div>'
        f'<div style="color:#aab3c5; font-size:.94rem; line-height:1.65;">{body}</div></div>',
        unsafe_allow_html=True,
    )


def animated_metrics(items: list[dict], height: int = 170) -> None:
    """Glass metric cards with animated count-up (rendered in a component iframe)."""
    cards = "\n".join(
        f"""
        <div class="m" style="animation-delay:{i * 0.08:.2f}s">
          <div class="mi">{it.get("icon", "")}</div>
          <div class="ml">{it["label"]}</div>
          <div class="mv" data-v="{it["value"]}" data-d="{it.get("decimals", 0)}" data-s="{it.get("suffix", "")}">0</div>
        </div>
        """
        for i, it in enumerate(items)
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"/><style>
    body{{margin:0;font-family:'Inter',system-ui,sans-serif;background:transparent;
      display:flex;gap:14px;padding:2px;flex-wrap:wrap;}}
    .m{{flex:1 1 150px;min-width:140px;background:rgba(255,255,255,.045);
      border:1px solid rgba(255,255,255,.09);border-radius:18px;padding:18px 20px;
      backdrop-filter:blur(14px);animation:r .7s cubic-bezier(.4,0,.2,1) both;
      transition:transform .3s,border-color .3s,box-shadow .3s;}}
    .m:hover{{transform:translateY(-3px);border-color:rgba(96,165,250,.4);
      box-shadow:0 14px 40px rgba(2,6,17,.5);}}
    @keyframes r{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:none}}}}
    .mi{{font-size:1.2rem;margin-bottom:8px;opacity:.9}}
    .ml{{color:#8f98ad;font-size:.7rem;font-weight:700;letter-spacing:.09em;
      text-transform:uppercase;margin-bottom:8px}}
    .mv{{color:#fff;font-size:1.75rem;font-weight:800;letter-spacing:-.01em}}
    </style></head><body>{cards}
    <script>
    const els=[...document.querySelectorAll('.mv')];
    els.forEach(el=>{{
      const v=parseFloat(el.dataset.v), d=parseInt(el.dataset.d), s=el.dataset.s;
      const t0=performance.now(), dur=1300;
      const fmt=(n)=>n.toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}})+s;
      (function tick(now){{
        const p=Math.min((now-t0)/dur,1), q=1-Math.pow(1-p,3);
        el.textContent=fmt(v*q);
        if(p<1)requestAnimationFrame(tick); else el.textContent=fmt(v);
      }})(performance.now());
    }});
    </script></body></html>"""
    st.iframe(html, height=height)


def _typing_hero(phrases: list[str]) -> None:
    phrases_js = json.dumps(phrases)
    html = f"""<!doctype html><html><head><meta charset="utf-8"/><style>
    body{{margin:0;background:transparent;display:flex;justify-content:center;
      align-items:center;min-height:100px;font-family:'Inter',system-ui,sans-serif;}}
    #tx{{color:#aab3c5;font-size:1.02rem;font-weight:500;}}
    #cr{{display:inline-block;width:2px;height:1.05em;background:#3b82f6;margin-left:3px;
      vertical-align:-.18em;animation:blink 1s step-end infinite;}}
    @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
    </style></head><body>
    <div><span id="tx"></span><span id="cr"></span></div>
    <script>
    const phrases={phrases_js};
    let pi=0,ci=0,del=false;
    const el=document.getElementById('tx');
    (function loop(){{
      const cur=phrases[pi];
      if(!del){{ ci++; if(ci>cur.length){{del=true; setTimeout(loop,1900); return;}} }}
      else {{ ci--; if(ci===0){{del=false; pi=(pi+1)%phrases.length;}} }}
      el.textContent=cur.slice(0,ci);
      setTimeout(loop, del?26:42);
    }})();
    </script></body></html>"""
    st.iframe(html, height=110)

# ---------------------------------------------------------------------------
# Nexora branding in the sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1.4rem 0 0.5rem 0;">
            <div style="font-size:2.85rem; font-weight:900; letter-spacing:0.06em;
                        background:linear-gradient(120deg,#60a5fa,#a78bfa,#22d3ee);
                        background-size:220% 220%;
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        animation:nx-grad-move 8s ease infinite;">
                NEXORA
            </div>
            <div style="color:#8f98ad; font-size:.82rem; margin-top:.2rem; font-weight:500;">
                AI That Connects Meaning
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

NAV = [
    "🏠 Home",
    "🌐 Live Translation",
    "🧠 Attention Explorer",
    "📊 Dataset & EDA",
    "🤖 Classical ML",
    "🕸️ Seq2Seq Neural Model",
    "📚 About / Conclusion",
]
page = st.sidebar.radio("Navigation", NAV, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Academic mini-project · Seq2Seq Networks with "
    "Attention-Based Alignment · © 2026 NEXORA"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "outputs" / "plots"
METRICS = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "models" / "seq2seq"


def _load_json(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def load_lid_df() -> pd.DataFrame:
    p = PROCESSED / "lid_dataset.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["text", "language"])


@st.cache_resource(show_spinner=False)
def load_pairs() -> dict:
    out = {}
    for name in ("hi", "mr"):
        p = PROCESSED / f"pair_{name}_sample.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


@st.cache_resource(show_spinner=False)
def load_stats() -> dict:
    return _load_json(PROCESSED / "dataset_stats.json") or {}


@st.cache_resource(show_spinner=False)
def load_classical_metrics() -> dict:
    return _load_json(METRICS / "classical_metrics.json") or {}


@st.cache_resource(show_spinner=False)
def load_seq2seq_metrics() -> dict:
    return _load_json(MODEL_DIR / "metrics.json") or {}


@st.cache_resource(show_spinner=False)
def get_translator():
    from src.translation import get_translator as _gt
    return _gt()


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def render_home():
    stats = load_stats()

    # ---- hero ----
    st.markdown(
        """
        <div style="text-align:center; padding:1.6rem 0 .6rem 0;">
            <div style="font-size:.78rem; font-weight:800; letter-spacing:.22em;
                        color:#93b4fb; text-transform:uppercase;">Neural Machine Translation</div>
            <div class="nx-grad-text" style="font-size:3.3rem; font-weight:900; letter-spacing:-.02em;
                        margin-top:.55rem; line-height:1.1;">NEXORA</div>
            <div style="font-size:1.15rem; color:#aab3c5; font-weight:500; margin-top:.4rem;">
                Sequence-to-Sequence Networks with Attention-Based Alignment
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _typing_hero([
        "Translate across 10 languages — English, Hindi, Marathi, Bengali, Gujarati…",
        "Watch real attention weights align source & target tokens…",
        "6 Indic scripts, 359,505 clean corpus rows, 3 classical baselines…",
        "Our own BiLSTM + GRU + Bahdanau attention — trained on CPU…",
    ])

    # ---- animated stat strip ----
    if stats:
        animated_metrics([
            {"icon": "📚", "label": "Corpus rows", "value": stats.get("raw_rows", 0)},
            {"icon": "🧹", "label": "Cleaned LID rows", "value": stats.get("lid_rows", 0)},
            {"icon": "🌍", "label": "Languages", "value": len(stats.get("languages", {}))},
            {"icon": "🤖", "label": "Classifiers", "value": 3},
            {"icon": "🧠", "label": "Attention model", "value": 1},
        ], height=158)
        st.write("")

    col1, col2 = st.columns([1.55, 1])
    with col1:
        nex_card(
            "What is Nexora?",
            "An end-to-end multilingual machine translation platform. It pairs a "
            "<b>pretrained production engine</b> (<code>facebook/m2m100_418M</code>, "
            "100 languages, 7.5B parallel sentences) with an <b>experimental "
            "Seq2Seq network with Bahdanau attention-based alignment</b> that we "
            "train in this project — and whose <b>real attention weights</b> you "
            "can explore as heatmaps.",
            icon="🌐", delay=0,
        )
        nex_card(
            "Why attention matters",
            "Classic encoder–decoder models compress the whole source sentence "
            "into one fixed vector. Attention lets the decoder <i>look back</i> at "
            "every source word at every step, producing an <b>alignment</b> between "
            "source and target tokens. This is the idea that turned neural machine "
            "translation from research into product.",
            icon="🧠", delay=1,
        )
        nex_card(
            "Project contents",
            "<b>Classical ML:</b> TF-IDF language identification with Naïve Bayes, "
            "Logistic Regression & SVM · <b>Neural:</b> BiLSTM encoder + GRU decoder "
            "+ additive attention, BLEU evaluation · <b>Live app:</b> 10-language "
            "translation with attention explorer.",
            icon="🗂️", delay=2,
        )
    with col2:
        if (PLOTS / "wordclouds.png").exists():
            st.image(str(PLOTS / "wordclouds.png"),
                     caption="Word clouds per language — generated from real corpus data",
                     width="stretch")
        else:
            st.info("Run `python build_pipeline.py` to generate EDA artifacts.")
        nex_card(
            "Dataset",
            f"<b>{DATASET_NAME}</b> — multilingual parallel corpus "
            f"(AI4Bharat, IIT Madras) · "
            f"<a href='{DATASET_URL}' style='color:#93b4fb;'>Hugging Face</a> · "
            "streamed subset, cleaned & deduplicated.",
            icon="📖", delay=3,
        )

    st.markdown("### Quick start")
    st.caption("Select languages and type text below — or head to **🌐 Live Translation**.")
    src = st.selectbox("Source language", list(LANGUAGES.keys()),
                       format_func=lambda k: f"{LANGUAGES[k]['flag']} {LANGUAGES[k]['name']}",
                       key="home_src")
    tgt = st.selectbox("Target language", list(LANGUAGES.keys()),
                       format_func=lambda k: f"{LANGUAGES[k]['flag']} {LANGUAGES[k]['name']}",
                       index=1, key="home_tgt")
    text = st.text_area("Text to translate", height=90, key="home_text",
                        placeholder="Type anything — in any of the 10 languages…")
    if st.button("✨ Translate", type="primary"):
        if text.strip():
            with st.spinner("Translating…"):
                try:
                    res = get_translator().translate(text.strip(), LANGUAGES[src]["m2m"], LANGUAGES[tgt]["m2m"])
                    st.success(res.target)
                except Exception as e:
                    st.error(f"Translation failed: {e}")
        else:
            st.warning("Please enter some text.")


# ---------------------------------------------------------------------------
# Live Translation
# ---------------------------------------------------------------------------

def render_translation():
    section_header("Live Translation",
                   "Production engine: facebook/m2m100_418M (pretrained multilingual model) · 10 supported languages",
                   kicker_text="⚡ Engine")

    col1, col2 = st.columns(2)
    with col1:
        src = st.selectbox("Source language", list(LANGUAGES.keys()),
                           format_func=lambda k: f"{LANGUAGES[k]['flag']} {LANGUAGES[k]['name']}", key=f"t_src")
    with col2:
        tgt = st.selectbox("Target language", list(LANGUAGES.keys()),
                           format_func=lambda k: f"{LANGUAGES[k]['flag']} {LANGUAGES[k]['name']}", index=1, key="t_tgt")

    swap_a, swap_b = st.columns([2.4, 4])
    with swap_a:
        if st.button("⇄ Swap languages", key="swap_lang"):
            st.session_state["t_src"], st.session_state["t_tgt"] = tgt, src
            st.rerun()

    if src == tgt:
        st.warning("Source and target are the same — pick different languages.")

    text = st.text_area("📝 Text to translate", height=120,
                        placeholder="e.g. How are you? · आप कैसे हैं? · Comment allez-vous ?")
    colb1, colb2 = st.columns([1, 3])
    with colb1:
        go = st.button("🚀 Translate", type="primary")
    with colb2:
        st.caption(f"Engine: **{MODEL_NAME}** · first load downloads ~1.9 GB (cached afterwards)")

    if go and text.strip() and src != tgt:
        with st.spinner("Running the model…"):
            try:
                res = get_translator().translate(text.strip(), LANGUAGES[src]["m2m"], LANGUAGES[tgt]["m2m"])
            except Exception as e:
                st.error(f"Translation error: {e}")
                return

        st.markdown("### Output")
        out1, out2 = st.columns(2)
        with out1:
            nex_card(f"{LANGUAGES[src]['flag']} Source ({LANGUAGES[src]['name']})",
                     text.strip().replace("<", "&lt;").replace(">", "&gt;"), icon="📥", delay=0)
        with out2:
            st.markdown(
                f"""
                <div class="nx-card" style="border:1px solid rgba(96,165,250,.45);
                            background:linear-gradient(135deg, rgba(59,130,246,.14), rgba(139,92,246,.14));">
                    <div style="font-weight:700; color:#93b4fb; margin-bottom:.5rem;
                                letter-spacing:.03em; text-transform:uppercase; font-size:.8rem;">
                        {LANGUAGES[tgt]['flag']} Translation · {LANGUAGES[tgt]['name']}
                    </div>
                    <div style="font-size:1.4rem; color:#fff; line-height:1.65; font-weight:600;">
                        {res.target}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with st.expander("🔎 Translation metadata"):
            st.json({
                "engine": res.engine,
                "source_lang": res.source_lang,
                "target_lang": res.target_lang,
                "input_length_chars": len(text.strip()),
                "output_length_chars": len(res.target),
                "note": "Pretrained model — NEXORA's own trained model is the experimental Seq2Seq (see Neural Model tab).",
            })

    if not (go and text.strip()):
        st.markdown("### Try an example")
        st.caption("One click: fills the selectors and text, then hit Translate.")
        examples = [
            ("How are you?", "en", "hi"),
            ("Nexora connects meaning across languages.", "en", "mr"),
            ("The weather is beautiful today.", "en", "bn"),
            ("What is your name?", "en", "ta"),
            ("Bonjour, comment allez-vous ?", "fr", "en"),
            ("आज बहुत सुंदर दिन है।", "hi", "en"),
        ]
        cols = st.columns(3)
        for i, (ex, s_, t_) in enumerate(examples):
            with cols[i % 3]:
                if st.button(f"“{ex}”  ({LANGUAGES[s_]['name']} → {LANGUAGES[t_]['name']})", key=f"ex_{i}"):
                    st.session_state[f"t_src"] = s_
                    st.session_state[f"t_tgt"] = t_
                    st.session_state["t_text"] = ex
                    st.rerun()


# ---------------------------------------------------------------------------
# Attention Explorer
# ---------------------------------------------------------------------------

def render_attention():
    section_header("Attention Explorer",
                   "Real attention-based alignment from the experimental Seq2Seq model we trained",
                   kicker_text="🧠 Alignment")

    metrics = load_seq2seq_metrics()
    if not metrics:
        st.warning("Run `python build_pipeline.py --steps seq2seq` first to train the experimental model.")
        return

    st.markdown(
        "This tab shows the **actual attention weights** computed by our trained "
        "encoder–decoder while translating a sentence. Rows are target tokens, "
        "columns are source tokens; brighter = stronger alignment."
    )

    tab_model, tab_live, tab_theory = st.tabs(["🖼️ Model samples", "⚡ Live demo", "📐 Theory"])

    with tab_model:
        st.markdown("#### Sample translations with real attention weights")
        samples = _load_json(MODEL_DIR / "sample_translations.json") or []
        attn = None
        attn_path = MODEL_DIR / "attention_sample.npy"
        if attn_path.exists():
            import numpy as np
            attn = np.load(attn_path)

        from src.multilingual import light_tokenize

        if samples and attn is not None:
            for i, row in enumerate(samples[:6]):
                src_toks = light_tokenize(row["source"])
                pred_toks = light_tokenize(row["prediction"])
                if not pred_toks:
                    continue
                w = attn[i, : len(pred_toks), : len(src_toks)]
                st.markdown(
                    f"""
                    <div class="nx-card" style="margin-top:.9rem;">
                        <div style="font-weight:700; color:#fff;">{i + 1}. {row['source']}</div>
                        <div style="color:#93b4fb; margin-top:.25rem; font-size:.92rem;">Prediction:
                        <span style="color:#e2e8f0;">{row['prediction']}</span></div>
                        <div style="color:#8f98ad; font-size:.86rem; margin-top:.15rem;">Reference:
                        {row['reference']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                from src.attention import tokens_heatmap
                fig = tokens_heatmap(w, src_toks, pred_toks,
                                     title="Attention-Based Alignment")
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("No attention samples yet — run the pipeline first.")

    with tab_live:
        st.markdown("#### Type a sentence — see how the model aligns tokens while translating it")
        live_src = st.text_input("English sentence", value="How are you ?", key="live_src")
        if st.button("🧠 Compute attention", type="primary", key="live_go"):
            with st.spinner("Running encoder–decoder with attention…"):
                try:
                    import torch
                    from src.seq2seq import Seq2SeqAttention, Vocab
                    model_state = torch.load(MODEL_DIR / "seq2seq_attention.pt", map_location="cpu")
                    vs = json.loads((MODEL_DIR / "vocab_src.json").read_text(encoding="utf-8"))
                    vt = json.loads((MODEL_DIR / "vocab_tgt.json").read_text(encoding="utf-8"))
                    v_src = Vocab(); v_src.stoi = vs["stoi"]; v_src.itos = vs["itos"]
                    v_tgt = Vocab(); v_tgt.stoi = vt["stoi"]; v_tgt.itos = vt["itos"]
                    model = Seq2SeqAttention(len(v_src), len(v_tgt), 128, 128)
                    model.load_state_dict(model_state)
                    model.eval()
                    preds, weights = model.translate([live_src], v_src, v_tgt, max_len=24)
                    src_toks = light_tokenize(live_src)
                    tgt_toks = light_tokenize(preds[0]) if preds[0] else ["<eos>"]
                    w = weights[0, : len(tgt_toks), : len(src_toks)]
                    st.markdown(
                        f"""
                        <div class="nx-card" style="border:1px solid rgba(96,165,250,.45);">
                            <div style="font-weight:700; color:#93b4fb; text-transform:uppercase;
                                        letter-spacing:.05em; font-size:.78rem;">Model translation</div>
                            <div style="font-size:1.25rem; color:#fff; font-weight:600; margin-top:.3rem;">
                                {preds[0]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    from src.attention import tokens_heatmap
                    fig = tokens_heatmap(w, src_toks, tgt_toks, title="Attention-Based Alignment")
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"Live attention failed: {e}")

    with tab_theory:
        from src.attention import alignment_explanation_card
        info = alignment_explanation_card()
        st.markdown("#### How attention-based alignment works")
        nex_card("Alignment question", info["what"], icon="❓", delay=0)
        st.latex(info["math"])
        nex_card("How to read the heatmap", info["reading"], icon="👁️", delay=1)


# ---------------------------------------------------------------------------
# Dataset & EDA
# ---------------------------------------------------------------------------

def render_dataset():
    section_header("Dataset & EDA",
                   "ai4bharat/samanantar — multilingual parallel corpus (AI4Bharat, IIT Madras) · real measured statistics",
                   kicker_text="📊 Data")

    stats = load_stats()
    lid_df = load_lid_df()
    pairs = load_pairs()

    if not stats or lid_df.empty:
        st.warning("Run `python build_pipeline.py` first to generate processed data and EDA figures.")
        return

    animated_metrics([
        {"icon": "📚", "label": "Raw corpus rows", "value": stats.get("raw_rows", 0)},
        {"icon": "🧹", "label": "Processed LID rows", "value": stats.get("lid_rows", 0)},
        {"icon": "🌍", "label": "Languages", "value": len(stats.get("languages", {}))},
        {"icon": "🧪", "label": "Test rows", "value": stats.get("splits", {}).get("test", 0)},
    ], height=150)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        nex_card("Dataset card",
                 f"<b>{DATASET_NAME}</b> · {DATASET_SOURCE} · "
                 f"columns: <code>{', '.join(COLUMNS)}</code>",
                 icon="📖", delay=0)
    with c2:
        nex_card("Cleanliness",
                 f"Missing: <b>{stats.get('missing_raw', {}).get('text', 0):,}</b> · "
                 f"Duplicates removed: <b>{stats.get('duplicates_raw', {}).get('text', 0):,}</b> · "
                 f"Split: train {stats.get('splits', {}).get('train', 0):,} / "
                 f"valid {stats.get('splits', {}).get('valid', 0):,} / "
                 f"test {stats.get('splits', {}).get('test', 0):,}",
                 icon="🧼", delay=1)

    st.markdown("### Dataset card")
    st.json({
        "name": DATASET_NAME,
        "source": DATASET_SOURCE,
        "url": DATASET_URL,
        "columns": COLUMNS,
        "rows": stats.get("raw_rows"),
        "missing_values": stats.get("missing_raw"),
        "duplicate_records": stats.get("duplicates_raw"),
        "train/valid/test": stats.get("splits"),
        "note": "Statistics measured from the streamed subset during preprocessing.",
    })

    st.markdown("### Exploratory visualizations")
    plots = {
        "language_distribution.png": "Language distribution",
        "length_distribution.png": "Sentence-length distribution",
        "src_vs_tgt_lengths.png": "Source vs target lengths (English→Hindi/Marathi)",
        "vocabulary.png": "Vocabulary size per language",
        "frequent_words.png": "Most frequent words (stopwords removed)",
        "quality.png": "Data quality — missing & duplicates",
        "wordclouds.png": "Word clouds per language (real data)",
    }
    for i, (fname, caption) in enumerate(plots.items()):
        p = PLOTS / fname
        if p.exists():
            st.image(str(p), caption=caption, width="stretch")

    csv_path = PLOTS / "dataset_stats.csv"
    if csv_path.exists():
        st.markdown("### Corpus statistics table")
        st.dataframe(pd.read_csv(csv_path), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Classical ML
# ---------------------------------------------------------------------------

def render_classical():
    section_header("Classical ML — Language Identification",
                   "TF-IDF / CountVectorizer features · Naïve Bayes · Logistic Regression · SVM",
                   kicker_text="🤖 Baselines")

    payload = load_classical_metrics()
    lid_df = load_lid_df()
    if not payload or lid_df.empty:
        st.warning("Run `python build_pipeline.py --steps classical` first.")
        return

    results = pd.DataFrame(payload["results"])

    st.markdown("### Task")
    st.info(
        "**Auxiliary task:** given a sentence in one of 6 languages, predict the "
        "language. Features come from character n-gram TF-IDF (script-specific "
        "fingerprints); the classifiers are trained on a stratified 80/10/10 split."
    )

    animated_metrics([
        {"icon": "🧮", "label": row["model"], "value": row["accuracy"] * 100,
         "decimals": 2, "suffix": "%", "mi": row["model"]}
        for row in payload["results"]
    ], height=158)
    st.write("")

    st.markdown("### Model comparison")
    st.dataframe(results.style.format(
        {c: "{:.4f}" for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]}
    ), width="stretch", hide_index=True)
    st.image(str(METRICS / "model_comparison.png"), caption="Test-set comparison", width="stretch")

    t_cm, t_rep, t_feat = st.tabs(["Confusion matrices", "Classification report", "Feature engineering"])

    with t_cm:
        for key in ("naive_bayes", "logistic_regression", "svm"):
            fig = METRICS / f"confusion_{key}.png"
            if fig.exists():
                st.image(str(fig), caption=key.replace("_", " ").title(), width="stretch")
    with t_rep:
        for key, label in (("naive_bayes", "Naïve Bayes"), ("logistic_regression", "Logistic Regression"), ("svm", "SVM")):
            st.markdown(f"**{label}**")
            st.text(payload["reports"][key])
    with t_feat:
        st.markdown(
            "**Why TF-IDF and CountVectorizer?** Classical models cannot consume "
            "raw text — they need numeric features. Word-level BoW counts and "
            "TF-IDF (term frequency × inverse document frequency) turn text into "
            "sparse vectors: TF-IDF down-weights frequent function words and boosts "
            "discriminative tokens. For language identification, character n-grams "
            "capture the *script fingerprint* — e.g. Devanagari conjuncts "
            "distinguish Hindi from Marathi."
        )
        st.latex(r"\text{tfidf}(t,d)=\text{tf}(t,d)\cdot\log\!\frac{1+N}{1+\text{df}(t)}+1")
        feat_files = list(PROCESSED.glob("features_*.npz"))
        st.caption(f"{len(feat_files)} feature matrices cached in `data/processed/`.")
        if lid_df is not None:
            st.caption(f"Feature vocabulary is derived from {len(lid_df):,} real sentences.")


# ---------------------------------------------------------------------------
# Seq2Seq Neural Model
# ---------------------------------------------------------------------------

def render_neural():
    section_header("Seq2Seq Neural Model",
                   "Encoder–Decoder with Bahdanau attention-based alignment — the model we actually trained",
                   kicker_text="🕸️ Neural")

    metrics = load_seq2seq_metrics()
    if not metrics:
        st.warning("Run `python build_pipeline.py --steps seq2seq` first to train the experimental model.")
        return

    st.markdown(f"""
    **Architecture:**
    * **Encoder** — BiLSTM over word embeddings → hidden states h₁…h_T
    * **Attention** — Bahdanau additive: e_tj = vᵀ·tanh(W·h_j + U·s_{{t−1}}),
      α = softmax(e), context c_t = Σ α_j·h_j
    * **Decoder** — GRU( [y_{{t−1}} ; c_t] ) → output projection → token
    """)

    animated_metrics([
        {"icon": "🎯", "label": "BLEU (test)", "value": metrics["bleu"]["bleu"],
         "decimals": 4},
        {"icon": "📉", "label": "Train loss", "value": metrics["history"]["train_loss"][-1],
         "decimals": 3},
        {"icon": "📈", "label": "Valid loss", "value": metrics["history"]["valid_loss"][-1],
         "decimals": 3},
        {"icon": "🧮", "label": "Vocab size", "value": metrics["vocab_src_size"]},
    ], height=158)
    st.write("")

    st.markdown("### Training curves (real)")
    hist = metrics["history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train loss", color="#3b82f6")
    ax.plot(hist["epoch"], hist["valid_loss"], marker="s", label="Valid loss", color="#8b5cf6")
    ax.set_facecolor("#0d1526")
    fig.patch.set_facecolor("#0d1526")
    ax.tick_params(colors="#aab3c5")
    ax.xaxis.label.set_color("#aab3c5")
    ax.yaxis.label.set_color("#aab3c5")
    ax.title.set_color("#e8ecf4")
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.18))
    legend = ax.legend()
    for t in legend.get_texts():
        t.set_color("#e8ecf4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Seq2Seq+Attention training history")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Sample translations (Source → Reference → Prediction → per-sample BLEU)")
    samples = _load_json(MODEL_DIR / "sample_translations.json") or []
    if samples:
        from src.evaluation import compute_bleu
        bleu_map = {}
        for i, s in enumerate(samples):
            b = compute_bleu([s["prediction"]], [s["reference"]], None)
            bleu_map[i] = f"{b['bleu']:.3f}"
        rows = []
        for i, s in enumerate(samples[:10]):
            rows.append({"Source": s["source"], "Reference": s["reference"],
                         "Prediction": s["prediction"], "BLEU": bleu_map[i]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("### Metric detail")
    st.json({
        "bleu": metrics["bleu"],
        "vocab_src": metrics["vocab_src_size"],
        "vocab_tgt": metrics["vocab_tgt_size"],
        "device": metrics["device"],
        "architecture": metrics["architecture"],
        "honesty": metrics["note"],
    })


# ---------------------------------------------------------------------------
# About / Conclusion
# ---------------------------------------------------------------------------

def render_about():
    section_header("About & Conclusion",
                   "Key findings · challenges · future scope · applications",
                   kicker_text="📝 Wrap-up")

    st.markdown("## Key findings")
    for i, f_ in enumerate([
        "Character n-gram TF-IDF achieves very high language-identification accuracy — script "
        "fingerprints are strong, simple discriminators.",
        "Seq2Seq with additive attention learns meaningful source–target alignment even with a "
        "small vocabulary and a few epochs on CPU.",
        "BLEU on the experimental model is modest (as expected for a bounded demo model) — the "
        "pretrained M2M100 engine gives production-grade output.",
        "Attention heatmaps reveal intuitive correspondences (names, numerals, frequent function "
        "words) and expose ambiguity when alignment spreads across many source tokens.",
    ]):
        nex_card("Finding", f_, icon="✅", delay=i)

    st.markdown("## Challenges")
    for i, f_ in enumerate([
        "Multilingual data availability — the Samanantar corpus has strong coverage for 6 Indic "
        "languages; wrong-alignment rows require script-consistency filtering.",
        "Vocabulary size and sequence length — morphologically rich Indic languages inflate "
        "vocabularies; we cap at 10k tokens with truncation.",
        "Training time — full corpora training is impractical on CPU; we train on a bounded sample.",
        "Translation ambiguity and rare words — scripts/false friends degrade attention alignment.",
    ]):
        nex_card("Challenge", f_, icon="⚠️", delay=i)

    st.markdown("## Future scope")
    for i, f_ in enumerate([
        "Transformer architecture (per-layer attention) instead of RNN-based attention",
        "Beam search decoding, length penalty, BLEU-tuned decoding",
        "Larger multilingual datasets (e.g. FLORES-200, WMT) and more languages",
        "Speech translation (ASR → MT) and document translation with segmentation",
        "Mobile/edge deployment with quantized models; human evaluation studies",
        "Domain-specific translation (medical, legal, education) fine-tuning",
    ]):
        nex_card("Roadmap", f_, icon="🚀", delay=i)

    st.markdown("## Applications")
    for i, f_ in enumerate([
        "Education — learning aids & transliteration for Indian languages",
        "Tourism & hospitality — real-time phrase translation",
        "Government services — multilingual public information",
        "Customer support & e-commerce — instant responses in the user's language",
        "Accessibility — text-to-speech pipelines for low-resource languages",
        "Content localization — subtitles, documentation, marketing",
    ]):
        nex_card("Use case", f_, icon="💡", delay=i)

    st.markdown("---")
    st.markdown(
        "**References** — Bahdanau et al. (2015) *Neural Machine Translation by "
        "Jointly Learning to Align and Translate* · Sutskever et al. (2014) "
        "*Sequence to Sequence Learning with Neural Networks* · Fan et al. (2021) "
        "*Beyond English-Centric Multilingual Machine Translation* (M2M-100) · "
        "Papineni et al. (2002) *BLEU* · Ramesh et al. (2022) *Samanantar* (AI4Bharat)."
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def main():
    if page.startswith("🏠"):
        render_home()
    elif page.startswith("🌐"):
        render_translation()
    elif page.startswith("🧠"):
        render_attention()
    elif page.startswith("📊"):
        render_dataset()
    elif page.startswith("🤖"):
        render_classical()
    elif page.startswith("🕸️"):
        render_neural()
    else:
        render_about()


if __name__ == "__main__":
    main()