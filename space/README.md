---
title: Lore Preview
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Lore Preview

Deterministic preview of Lore's wiki shape: ingest one source, inspect the generated page, browse the vault, and see the wikilink graph.

This Space is intentionally lighter than the full local Lore workflow:

- seeds a writable starter wiki
- ingests one URL or uploaded PDF/Markdown source
- writes a single deterministic summary page into `wiki/`
- lets you inspect the page list and link graph directly
- suggests what to explore next with heuristics instead of a live checkpoint

For the full product story, run Lore locally with Claude Code or MCP so the agent can keep maintaining the wiki over time.

Use the repo's `scripts/deploy_space.sh` helper to build a self-contained Space bundle and upload it with the Hugging Face CLI.
