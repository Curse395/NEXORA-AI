# 🚀 NEXORA — Deployment Guide

> How to put this app live so **anyone** can access it, with realistic
> expectations about the heavy M2M100 model.

---

## TL;DR

> ⚠️ **Important (2025+):** Hugging Face changed its free tier — **only
> *static* Spaces are free**. Gradio / Docker (incl. Streamlit) Spaces on the
> free `cpu-basic` hardware now require a **PRO subscription** ($9/mo). See
> [Option A](#option-a--hugging-face-spaces-streamlit) for details.

| Option | Cost | Live M2M100 translation? | Effort |
|---|---|---|---|
| **B. Streamlit Community Cloud** | Free (1 GB RAM) | ⚠️ App runs fully; Translation tab shows graceful "not enough memory" (M2M100 guarded) | ⭐ easiest, limited |
| **A. Hugging Face Spaces (Streamlit)** | Free only for static; Streamlit/Docker needs **PRO** | ✅ Yes (lazy load, ~2–4 GB RAM) | ⭐⭐ |
| **C. Docker on VPS / Render / Railway** | ~$0–7/mo | ✅ Yes | ⭐⭐ |
| **D. GitHub Pages / static preview** | Free | ❌ No server — but gives a public URL that links to the repo + screenshots | ⭐⭐⭐ (not a real app) |

**Recommendation:** **B** (Streamlit Community Cloud) is the best **free**
option that runs the full interactive app — everything works, and we added a
memory guard so the Translation tab fails *gracefully* instead of OOM-ing on
the 1 GB free tier. **A** (HF Spaces) gives you live M2M100 translation but
now costs **PRO**. **C** if you want an always-on, controllable deployment.

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

## Option B — Streamlit Community Cloud ★ (best free)

Free, connects straight to your GitHub repo, runs the full interactive app.
The only limitation: the live M2M100 translation engine needs ~2–4 GB RAM,
and the free plan gives 1 GB — so we added a **memory guard** that detects
this and shows a friendly message instead of crashing.

1. Push the repo to **GitHub** (already done for this project):
   ```bash
   git init
   git add -A
   git commit -m "NEXORA app"
   git branch -M main
   git remote add origin https://github.com/<you>/nexora.git
   git push -u origin main
   ```
2. Go to **<https://share.streamlit.io>**, sign in with GitHub:
   - **New app** → pick the `owner/nexora` repo
   - Branch: `main` · Main file: `app.py`
3. Streamlit auto-deploys and gives a URL like `https://nexora.streamlit.app`.

> ✅ Everything works: Home, Dataset, Classical ML, Seq2Seq, Attention
> explorer, About.
> ⚠️ Pressing **Translate** on the free tier will raise a friendly
> "not enough memory" notice (the app detects < 1.8 GB free RAM and refuses
> to load the 1.9 GB model). Upgrade the plan or use Option A/C for live
> translation.

---

## Option A — Hugging Face Spaces (Streamlit)

> **2025+ reality check:** Hugging Face now requires **PRO** to host Gradio
> or **Docker** Spaces (Streamlit Spaces are Docker-backed) on the free
> `cpu-basic` hardware. Creating one from the CLI returns `402 Payment
> Required` with: *"Static Spaces are free for everyone, but hosting Gradio
> and Docker Spaces on free cpu-basic requires a PRO subscription."*
> Only **static** Spaces are free — which can't run Python/Streamlit.

With PRO, the process is identical to before (and worth it for live M2M100):
1. **Create a Space** — <https://huggingface.co/new-space>
   - Name: `nexora-mt` · SDK: **Streamlit** · Hardware: CPU basic (16 GB RAM)
2. **Push the code**
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/nexora-mt
   git push space main
   ```
3. Enable **Persistent Storage** on the Space (Settings) so the ~1.9 GB
   M2M100 cache survives restarts.

> The `README_hf.md` in this repo is a ready-made Space card (title, emoji,
> colors). Rename it to `README.md` in the Space and re-push for the card.

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
| Streamlit Cloud free | $0 | ⚠️ limited by 1 GB (guarded, friendly notice) | `nexora.streamlit.app` |
| HF Spaces CPU basic | $0 static / **PRO** for Streamlit+Gradio+Docker | ✅ (PRO) | `hf.co/spaces/you/nexora` |
| Render 2 GB instance | ~$5 | ✅ | `nexora.onrender.com` |
| Railway + volume | ~$5 | ✅ | `nexora.up.railway.app` |
| VPS (2 GB droplet) | ~$6 | ✅ | your IP/domain |
| GitHub Pages | $0 | ❌ static only | `you.github.io/nexora` |

**Bottom line:** start with **Streamlit Community Cloud** — free, zero setup,
and the whole app works live (the Translation tab shows a graceful notice on
the 1 GB free tier). If you want the real M2M100 Translate button, use **HF
Spaces with PRO** (16 GB RAM) or a **~$5 VPS/container**.