---
title: Lore Demo
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Lore Demo

Ingest one source, write one wiki page, and get three next questions.

This Space uses Lore's deterministic demo workflow instead of a heavyweight model checkpoint:

- seeds a writable starter wiki
- ingests one URL or uploaded PDF/Markdown source
- writes a single summary page into `wiki/`
- rebuilds the index and backlinks
- suggests what to explore next

Use the repo's `scripts/deploy_space.sh` helper to build a self-contained Space bundle and upload it with the Hugging Face CLI.
