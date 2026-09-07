---
title: NEXORA — MT with Attention
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.63.0
app_file: app.py
pinned: false
---
# NEXORA — Machine Translation with Seq2Seq + Attention

**AI That Connects Meaning** · Multilingual MT | Real attention heatmaps |
TF-IDF classical baselines | Self-trained Seq2Seq+Attention model.

Visit: <https://huggingface.co/spaces> → new Space → SDK **Streamlit** →
clone this repo → the app boots.

> **Memory note:** the live translation tab uses `facebook/m2m100_418M`
> (~1.9 GB download). Streamlit Spaces with a free CPU basic instance may be
> tight; the model loads lazily only when you press Translate. Everything else
> (attention explorer, dataset, classical ML, Seq2Seq page) works without it.