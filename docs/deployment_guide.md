# 🚀 NEXORA — Deployment Guide

> How to put this app live so **anyone** can access it, with realistic
> expectations about the heavy M2M100 model.

---

## TL;DR

| Option | Cost | Live M2M100 translation? | Effort |
|---|---|---|---|
| **A. Hugging Face Spaces (Streamlit)** | Free (CPU basic: 2 vCPU / 16 GB RAM) | ✅ Yes (lazy load, ~2–4 GB RAM while translating) | ⭐ easiest |
| **B. Streamlit Community Cloud** | Free (1 GB RAM) | ⚠️ Everything works *except* M2M100 may OOM | ⭐ easiest, limited |
| **C. Docker on VPS / Render / Railway** | ~$0–7/mo | ✅ Yes | ⭐⭐ |
| **D. GitHub Pages / static preview** | Free | ❌ No server — but gives a public URL that links to the repo + screenshots | ⭐⭐⭐ (not a real app) |

**Recommendation:** **A** if you want the full live translation experience for
free. **C** if you want an always-on, controllable deployment. **B** if you
don't mind the translation tab being limited.

---

## What the app needs (measured)

| Resource | Amount | Notes |
|---|---|---|
| Python | 3.10+ | dev was on 3.10.11 |
| Disk (app artifacts) | ~2.5 MB committed | parquets 36 MB, seq2seq .pt 21 MB — can be rebuilt |
| M2M100 model download | **~1.9 GB** | happens on first Translate press, cached after |
| RAM for M2M100 inference | **~2–4 GB** | CPU inference of the 418M model |
| RAM for everything else | < 1 GB | parquet read + small torch model |

Because M2M100 is downloaded **lazily** (only when you press Translate) and is
**cached in a volume**, you can deploy free of charge and only pay the memory
cost at the moment of translation.

---

## Option A — Hugging Face Spaces (recommended)

Hugging Face gives every Space a public URL and 16 GB RAM on the free CPU
"basic" tier — comfortably enough for M2M100.

1. **Create a Space**
   - Go to <https://huggingface.co/new-space>
   - Name: `nexora` (or similar)
   - SDK: **Streamlit**
   - Hardware: keep **CPU basic** (free: 2 vCPU, 16 GB RAM)
2. **Push the code**
   - In the project root:
     ```bash
     git init                     # if you haven't already
     git add -A
     git commit -m "NEXORA app"
     git remote add space https://huggingface.co/spaces/<your-username>/nexora
     git push space main
     ```
   - Hugging Face builds automatically (it reads `requirements.txt`, runs
     `streamlit run app.py`).
3. **Model cache (recommended)**
   - Spaces has a **Persistent Storage** toggle (Settings → Persistent Storage,
     50 GB free under the storage quota) — enable it and the 1.9 GB M2M100
     cache survives restarts. Without it, every cold start re-downloads.
4. **Done** — your app is live at
   `https://huggingface.co/spaces/<your-username>/nexora`

> The `README_hf.md` in this repo is a ready-made Space card (title, emoji,
> colors). If your Space card looks plain, copy its frontmatter into
> `README.md` and re-push.

---

## Option B — Streamlit Community Cloud

1. Push the repo to **GitHub**:
   ```bash
   git init
   git add -A
   git commit -m "NEXORA app"
   git branch -M main
   git remote add origin https://github.com/<you>/nexora.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, sign in with GitHub, **New app**,
   point at `owner/nexora`, branch `main`, file `app.py`.
3. Streamlit auto-deploys and gives a URL like
   `https://nexora.streamlit.app`.

> ⚠️ **RAM reality check:** free Streamlit Cloud offers **1 GB RAM**. The
> static pages (Home, Dataset, Classical ML, Seq2Seq, About, Attention
> samples) will run fine. Pressing **Translate** will try to load M2M100
> (~2–4 GB) and will likely be killed by the memory limit.
> **Workaround:** a paid plan, or **Option A/C** instead.

---

## Option C — Docker on Render / Railway / a VPS

The repo includes a ready-made `Dockerfile`.

### Local test first

```bash
docker build -t nexora .
docker run --rm -p 8501:8501 -v nexora-m2m:/app/models/m2m100 nexora
# open http://localhost:8501
```

The volume `nexora-m2m` caches M2M100 so rebuilds don't re-download.

### Render.com (free tier starts, paid ~$5/mo)

1. New **Web Service** → connect the GitHub repo.
2. Render detects the `Dockerfile` automatically.
3. Set:
   - Port: `8501`
   - Instance type: at least 2 GB RAM (free tier has 512 MB — enough for the
     dashboard, not for M2M100 inference; a $5 instance is recommended).
4. Deploy → public URL like `https://nexora.onrender.com`.

### Railway.app (~$5/mo after trial)

1. New Project → Deploy from GitHub repo.
2. Railway auto-detects the Dockerfile.
3. Add a Volume mounted at `/app/models/m2m100` (persists the model cache).
4. Set public domain → URL.

### Any Linux VPS (e.g. $4–6/mo DigitalOcean droplet)

```bash
# inside the droplet
docker build -t nexora .
docker run -d --name nexora -p 8501:8501 \
  -v nexora-m2m:/app/models/m2m100 --restart unless-stopped nexora
```

Then point a domain (or the droplet IP) at port 8501, optionally behind
Caddy/nginx.

---

## Option D — "Public link" without a server (demo only)

If you just want a URL to share for review (not a working app):

1. Push to GitHub (+ enable **GitHub Pages** on the repo).
2. That hosts the README + `docs/` + the screenshots in
   `outputs/screenshots/` at `https://<you>.github.io/nexora/`.
3. It is **static** — the ML app itself is not executable there.

This is useful as a *submission artifact link*, not a live app replacement.

---

## Before you push (checklist)

- [ ] `requirements.txt` installs cleanly on a fresh machine
      (`pip install -r requirements.txt`).
- [ ] `build_pipeline.py` artifacts exist OR you accept the "run pipeline
      first" stubs on the server. For Spaces/Render, either:
      - commit `data/processed/*.parquet`, `outputs/metrics/*`,
        `outputs/plots/*`, `models/seq2seq/*` (exclude the 547 MB `.npz`
        feature matrices — they are only needed to *retrain*), **or**
      - run `python build_pipeline.py --steps preprocess,eda,classical`
        once inside the deployed container (takes ~10–15 min on CPU).
- [ ] `.streamlit/config.toml` has `headless = true` (it does).
- [ ] For Spaces: enable **Persistent Storage** to cache M2M100.
- [ ] Test the deployed URL once with all 7 pages.

> **Minimum files for a working deployment** (everything else is optional on
> the server):
> ```
> app.py  src/  requirements.txt  .streamlit/config.toml
> data/processed/lid_dataset.parquet  dataset_stats.json
> outputs/metrics/classical_metrics.json  outputs/plots/*.png
> models/seq2seq/seq2seq_attention.pt  vocab_src.json  vocab_tgt.json
> models/seq2seq/metrics.json  sample_translations.json  attention_sample.npy
> ```

---

## Cost summary

| Path | Monthly cost | M2M100 works | URL example |
|---|---|---|---|
| HF Spaces CPU basic | $0 | ✅ | `hf.co/spaces/you/nexora` |
| Streamlit Cloud free | $0 | ⚠️ limited by 1 GB | `nexora.streamlit.app` |
| Render 2 GB instance | ~$5 | ✅ | `nexora.onrender.com` |
| Railway + volume | ~$5 | ✅ | `nexora.up.railway.app` |
| VPS (2 GB droplet) | ~$6 | ✅ | your IP/domain |
| GitHub Pages | $0 | ❌ static only | `you.github.io/nexora` |

**Bottom line:** start with **Hugging Face Spaces + Persistent Storage** — it's
free, 16 GB RAM, and purpose-built for exactly this stack.