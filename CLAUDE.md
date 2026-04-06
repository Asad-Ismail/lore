# Personal ML Research Knowledge Wiki

## Overview

A Karpathy-style personal knowledge base for ML/quantization research.
Two loops run simultaneously:
1. **Wiki loop**: `raw/ → /lore absorb → wiki/` — knowledge accumulates as markdown
2. **RL loop**: `/lore query → trajectory → GRPO reward → LoRA` — knowledge bakes into model weights

## Key Commands

| Command | What it does |
|---|---|
| `/lore ingest <path\|url>` | Add a paper, article, or note to `raw/` |
| `/lore absorb` | Compile all unprocessed raw/ sources into wiki articles |
| `/lore query <question>` | Answer a question using the wiki; captures training trajectory |
| `/lore status` | Show article counts, index freshness, RL training queue |
| `/lore rebuild-index` | Rebuild search index after bulk edits |
| `/lore health` | Find contradictions, stubs, orphans, undiscovered connections |
| `/lore cleanup` | Fix broken wikilinks, merge duplicates |
| `/lore render report <topic>` | Produce a long-form markdown research report |
| `/lore render slides <topic>` | Produce a Marp slide deck |
| `/lore reorganize` | Reclassify articles by emergent taxonomy |

## Architecture

```
raw/            ← source documents (never edit)
  papers/       ← PDFs, arXiv markdown
  articles/     ← Obsidian Web Clipper exports
  repos/        ← README + key files from GitHub repos
  images/       ← downloaded images
  notes/        ← freeform notes

wiki/           ← LLM-compiled wiki (Obsidian vault root)
  concepts/     ← core ML/quant ideas
  techniques/   ← specific methods (GPTQ, AWQ, LoRA, etc.)
  papers/       ← one article per paper
  models/       ← architecture summaries
  datasets/     ← dataset provenance
  benchmarks/   ← eval benchmark details
  people/       ← researcher profiles
  meta/         ← reading lists, research agendas

outputs/        ← generated artifacts (reports, slides, charts)
data/           ← all persistent state (SQLite, TF-IDF index, LoRA adapters)
```

## Python Tools (installed via `pip install -e .`)

- `lore-ingest <path>` — CLI for ingesting files
- `lore-search <query>` — CLI for searching the wiki
- `lore-absorb` — CLI for compilation
- `lore-rebuild-index` — CLI for index rebuild
- `lore-health` — CLI for health checks
- `lore-render report|slides <topic>` — CLI for output generation
- `lore-train status|train|rollback` — CLI for RL training management

## Models (from /home/ec2-user/SageMaker/hf_cache/)

- **Qwen/Qwen3-4B**: Primary inference (wiki compilation, Q&A, health checks)
- **Qwen/Qwen3-1.7B**: LoRA training target (evolving agent)
- **microsoft/Florence-2-base**: Image captioning for raw/images/

## Important Notes

- `raw/` is source-of-truth — never edit files there
- `wiki/` is LLM-maintained — you rarely need to edit manually; if you do, mark the edit in frontmatter
- `data/` is gitignored by default (SQLite dbs, index pkl, LoRA weights)
- Every `/lore query` automatically captures a training trajectory
- LoRA training triggers automatically after every 10 new trajectories
- The wiki follows [[WikiLink]] syntax — fully compatible with Obsidian

## Setup

```bash
bash scripts/setup.sh   # install deps, verify GPU + models
pip install -e .        # install CLI entry points
```

## Obsidian

Open `wiki/` as your Obsidian vault (`File > Open Vault > wiki/`).
The `_index.md` and `_summaries.md` files are auto-generated — don't edit them.
