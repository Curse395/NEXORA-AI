# 🌐 NEXORA — AI That Connects Meaning

**Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment**

Nexora is a complete, submission-ready NLP mini-project: a polished multilingual
machine-translation web application (Streamlit) plus a fully reproducible
academic pipeline covering classical ML baselines, a self-trained experimental
**Seq2Seq + Bahdanau attention** model, and **real attention-alignment
heatmaps**.

---

## ✨ Features

| Area | What you get |
|---|---|
| **Live translation** | 10 languages (English, Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, French, Spanish, German) via the pretrained `facebook/m2m100_418M` production engine |
| **Attention Explorer** | Real attention-weight heatmaps from our *own* trained encoder–decoder model (rows = target tokens, columns = source tokens) |
| **Dataset & EDA** | Full visual EDA of the Samanantar corpus — sentence-length distributions, vocabulary stats, most frequent words, word clouds |
| **Classical ML** | Language identification with TF-IDF + CountVectorizer, trained with Naïve Bayes, Logistic Regression and SVM; confusion matrices, ROC curves, comparison charts |
| **Seq2Seq neural model** | BiLSTM encoder + GRU decoder + additive attention, trained on an English→Hindi sample; training curves, BLEU, sample translations |
| **Documentation** | Full academic documentation + concise handwritten-theory notes |
| **Premium UI** | Dark glassmorphism design system, animated count-up metrics, typing hero, smooth hover/entrance animations |

---

## 🏗️ Architecture

```
                 ┌──────────────────────────────┐
   source text → │ PRODUCTION ENGINE (M2M100)   │ → translation
                 └──────────────────────────────┘
                    pretrained, 100 languages

                 ┌──────────────────────────────┐
   source text → │ EXPERIMENTAL Seq2Seq+Attn    │ → translation + real
                 │ Encoder(BiLSTM)              │   attention alignment
                 │ Decoder(GRU) + Bahdanau attn │   heatmaps
                 └──────────────────────────────┘
                    trained in this project (bounded sample)
```

The app clearly distinguishes the two:
1. **Production translation** — `facebook/m2m100_418M` (pretrained; we do **not** claim to have trained it).
2. **Experimental Seq2Seq + Attention** — the model *we actually train* for the academic requirement.

---

## 📊 Dataset

| Field | Value |
|---|---|
| **Name** | Samanantar — The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages |
| **Source** | Hugging Face Hub — `ai4bharat/samanantar` |
| **URL** | https://huggingface.co/datasets/ai4bharat/samanantar |
| **Columns** | `idx`, `src` (English), `tgt` (target language) |
| **Languages used** | Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu |
| **Reference** | Ramesh et al., *Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages*, TACL 2022 |

All corpus statistics shown in the dashboard (row counts, missing values,
duplicates, vocab, splits) are **measured from the actual data** during
preprocessing — nothing is hard-coded.

---

## 📦 Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

> The requirements pin versions that work together; if you use the exact
> versions listed, `pip` will resolve them. The first run of the translation
> tab also downloads the ~1.9 GB M2M100 model (cached locally afterwards).

---

## 🚀 Running locally

```bash
# 1. (Optional) pre-download the production model
python scripts/download_model.py

# 2. Build everything: preprocess → EDA → features → classical ML → seq2seq
python build_pipeline.py

# 3. Launch the dashboard
streamlit run app.py
```

`build_pipeline.py` writes reproducible artifacts:

```
data/processed/   lid_dataset.parquet, pair_hi_sample.parquet, pair_mr_sample.parquet, dataset_stats.json
outputs/plots/    *.png (EDA figures), dataset_stats.csv
outputs/metrics/  classical_results.csv, classical_metrics.json, confusion_*.png, roc_curves.png, model_comparison.png
models/seq2seq/   seq2seq_attention.pt, vocab_*.json, metrics.json, sample_translations.json, attention_sample.npy
```

> On very constrained hardware, `python build_pipeline.py --quick` uses a
> smaller corpus and 2 training epochs. A GPU is **not** required — the
> experimental model trains on CPU in a few minutes.

---

## 🧠 Model details

### Classical ML (auxiliary task: language identification)
- **Features:** CountVectorizer (BoW, word + char n-grams) and TF-IDF (word + char n-grams).
- **Models:** Multinomial Naïve Bayes, Logistic Regression (L2), Linear SVM.
- **Evaluation:** accuracy, macro precision/recall/F1, confusion matrices, one-vs-rest ROC.
- All metrics computed on a held-out test split (80/10/10).

### Experimental neural model (the academic core)
- **Encoder:** bi-directional LSTM over learned word embeddings → encoder hidden states.
- **Attention:** Bahdanau (additive) attention — scores, softmax weights, context vector.
- **Decoder:** GRU conditioned on the attention context, output projection, greedy token generation.
- **Training:** cross-entropy, teacher forcing, small bounded sample (English→Hindi).
- **Evaluation:** corpus BLEU (smooth, 1–4 gram) plus training/validation loss curves.

### Production engine
- `facebook/m2m100_418M` — Meta AI's 418M-parameter multilingual model
  (Fan et al., 2021), 100 languages. Used only for live high-quality
  translation; not retrained here.

---

## 📈 Evaluation

- **Classical:** per-model accuracy / precision / recall / F1 + confusion matrices + ROC curves, compared in a chart.
- **Neural:** BLEU score on held-out samples, train/valid loss curves, and a
  source–reference–prediction table with per-sample BLEU.
- The metrics JSON files under `outputs/metrics/` and `models/seq2seq/` are the
  single source of truth — they are generated, never hand-typed.

---

## 🖼️ Screenshots

> The live dashboard (dark premium theme) at `http://localhost:8501`:
> 1. Home — animated hero, typing tagline, glassmorphism metric counters
> 2. Live Translation — language swap, glass source/target cards, example chips
> 3. Attention Explorer — real heatmaps + live per-sentence alignment
> 4. Dataset & EDA — animated stat strip + all 7 EDA figures
> 5. Classical ML — animated accuracy cards, comparison chart, confusion matrices
> 6. Seq2Seq — animated BLEU/loss cards, dark-theme training curves, samples
> 7. About — key findings, challenges, roadmap, applications as glass cards

Captured captures live under `outputs/screenshots/`.

---

## ⚠️ Limitations

- The experimental Seq2Seq model is trained on a **bounded sample** (not the
  whole corpus) and uses a small vocabulary; BLEU is modest by design. It exists
  to demonstrate the *mechanism* (encoder, decoder, embeddings, hidden states,
  attention weights, alignment) with honest numbers.
- Live translation is **offline-dependent**: the first run downloads M2M100.
  Without the download, the translation tab shows a clear message instead of a
  fake result.
- Attention alignment is only available from our experimental model; M2M100's
  internal attention is not exposed.

---

## 🔭 Future scope

Transformer architectures, beam search, larger multilingual datasets (FLORES-200,
WMT), more languages, speech & document translation, mobile/edge deployment,
human evaluation, domain-specific fine-tuning.

---

## 📚 Documentation

- `docs/project_documentation.md` — full academic report (theory + results).
- `docs/application_guide.md` — **the complete app walkthrough**: every page,
  every component, where every number comes from, architecture, and FAQ.
- `docs/deployment_guide.md` — **how to deploy this live** (Hugging Face
  Spaces, Streamlit Cloud, Docker/Render/Railway/VPS + cost table).
- `docs/handwritten_notes.md` — concise notes designed for handwritten A4 pages.
- `notebooks/experiments.ipynb` — runnable experiment walkthrough.

---

## 🚀 Deploying live

> **2025+:** Hugging Face now requires **PRO** to host Streamlit/Gradio/Docker
> Spaces on free `cpu-basic` (only static Spaces are free). The best **free**
> way to run the full interactive app is **Streamlit Community Cloud**.

### Quickest free path — Streamlit Community Cloud (2 min)
1. This repo is already on GitHub (`Curse395/NEXORA-AI`).
2. Go to **<https://share.streamlit.io>** → sign in with GitHub → **New app**.
3. Pick the repo, branch `main`, main file **`app.py`** → **Deploy**.
4. URL: `https://nexora.streamlit.app` (auto).

> On the free 1 GB tier the whole app works; the live M2M100 Translate button
> is memory-guarded and shows a friendly notice instead of crashing.

### If you have HF PRO — Spaces (16 GB RAM, live M2M100)
```bash
git remote add space https://huggingface.co/spaces/<your-username>/nexora
git push space main
```
Then enable **Persistent Storage** in Space Settings so the ~1.9 GB M2M100
cache survives restarts.

Full instructions for all options (Streamlit Cloud, Spaces, Docker on
Render/Railway/VPS, static GitHub Pages) + a cost table:
→ **[docs/deployment_guide.md](docs/deployment_guide.md)**

---

## 📄 License / attribution

- Samanantar dataset © AI4Bharat (IIT Madras), distributed under the
  [CC BY 4.0-style terms](https://indicnlp.ai4bharat.org/samanantar/).
- M2M100 © Meta AI — model weights subject to their license.
- Project code: academic use for a university mini-project.

---

**NEXORA — AI That Connects Meaning**