---
title: NEXORA — MT with Attention
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---
# NEXORA — Machine Translation with Seq2Seq + Attention

**AI That Connects Meaning** · Multilingual MT | Real attention heatmaps |
TF-IDF classical baselines | Self-trained Seq2Seq+Attention model.

**Try it live:** the app is deployed on **Streamlit Community Cloud** from
this GitHub repo → <https://nexora.streamlit.app> (or via your fork).

> **HF Spaces note (2025+):** Hugging Face now requires **PRO** to host
> Streamlit/Gradio/Docker Spaces on free `cpu-basic` — only *static* Spaces
> are free. If you're on PRO, create a new Space (SDK **Streamlit**), push
> the repo, and this file becomes its card. On a **free** plan, use
> **Streamlit Community Cloud** instead — everything runs except the live
> M2M100 Translate button (memory-guarded with a friendly notice on 1 GB).

> **Memory note:** the live translation tab uses `facebook/m2m100_418M`
> (~1.9 GB download, ~2–4 GB RAM at inference). It loads lazily only when
> you press Translate, and the app now *detects* low-memory hosts first and
> fails fast with a friendly message. Everything else (attention explorer,
> dataset, classical ML, Seq2Seq page) works without it.