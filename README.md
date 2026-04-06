```
 _
| |    ___  _ __ ___
| |   / _ \| '__/ _ \
| |__| (_) | | |  __/
|_____\___/|_|  \___|
```

# Lore

**A personal ML research knowledge base that gets smarter every time you use it.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/you/lore?style=social)](https://github.com/you/lore)
[![Papers](https://img.shields.io/badge/arxiv-papers-red)](https://arxiv.org)

---

## The Problem

You read 200 papers a year. Six months later you can't remember the tradeoff between GPTQ and AWQ. You've tried Obsidian (great for writing, terrible for retrieval), Notion AI (cloud-dependent, no fine-tuning), and plain RAG (answers questions but never learns). None of them compound — the 200th paper you read doesn't make the system better at answering questions about the first 50.

Lore is different. It runs two loops simultaneously.

The **wiki loop** compiles every paper, article, and repo you feed it into a structured Markdown wiki using Qwen3-4B. Knowledge accumulates as interconnected articles with `[[WikiLink]]` citations, visible in Obsidian. This is your externalized memory.

The **RL loop** captures every question you ask as a training trajectory. A smaller model — Qwen3-1.7B — fine-tunes itself on those trajectories via GRPO, rewarded for grounding, citation precision, completeness, and fluency. The model that answers your 500th question has been shaped by the 499 questions you asked before it. It bakes your wiki's knowledge into its weights.

Both loops run locally. No API calls. No data leaves your machine.

---

## Demo

```
$ lore query "What is the key tradeoff between GPTQ and AWQ?"

  Searching wiki... 12 chunks retrieved (0.31s)
  Generating answer (Qwen3-1.7B + LoRA step-002400)...

  ┌─────────────────────────────────────────────────────────────────────┐
  │  GPTQ and AWQ represent two philosophically different approaches to  │
  │  post-training quantization, and the tradeoff is fundamentally about │
  │  where you do the work: at calibration time or at inference time.    │
  │                                                                      │
  │  GPTQ [[gptq-paper]] uses second-order information (the Hessian of  │
  │  each layer's reconstruction error) to find weight perturbations that│
  │  minimize output error under INT4 quantization. This is expensive    │
  │  upfront — calibration on a 70B model can take 4+ hours on a single  │
  │  GPU — but the resulting quantized weights are static and fast.      │
  │  [[post-training-quantization]]                                      │
  │                                                                      │
  │  AWQ [[awq-paper]] takes a different observation: not all weights    │
  │  matter equally. ~1% of weights are "salient" — they correspond to   │
  │  channels with high activation magnitude. AWQ protects these weights │
  │  by searching for a per-channel scale that minimizes quantization    │
  │  error on salient channels specifically. Calibration is far cheaper  │
  │  (minutes, not hours) and the method generalizes better across       │
  │  calibration datasets. [[activation-aware-quantization]]             │
  │                                                                      │
  │  The practical tradeoff: GPTQ achieves marginally better perplexity  │
  │  at INT4 on benchmarks where the calibration distribution matches    │
  │  the test distribution. AWQ is more robust out-of-distribution,      │
  │  faster to calibrate, and tends to be the community default for      │
  │  serving. For INT2/INT3, GPTQ's Hessian-guided rounding usually      │
  │  wins. [[quantization-benchmarks]]                                   │
  └─────────────────────────────────────────────────────────────────────┘

  Sources: gptq-paper, awq-paper, post-training-quantization,
           activation-aware-quantization, quantization-benchmarks
  Reward:  grounding=0.91  citation=0.83  coverage=0.88  fluency=0.79
  Total:   0.868  |  Trajectory saved → data/trajectories.db (#247)
  Model:   Qwen3-1.7B + LoRA step-002400  (trained on 247 trajectories)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/you/lore && cd lore
bash scripts/setup.sh && pip install -e .

# 2. Feed it a paper
lore ingest https://arxiv.org/abs/2306.00978   # AWQ paper

# 3. Compile into wiki articles
lore absorb

# 4. Ask your first question
lore query "How does AWQ decide which weights are salient?"
```

That's it. After 10 queries, Lore prints a retrain suggestion — run `lore-train train` when you're ready.

---

## Web UI

Lore ships a local web interface — dark research terminal aesthetic, no cloud.

```bash
lore-web              # → http://localhost:7860
lore-web --port 8080  # custom port
```

**Three panels:**

| Panel | What you can do |
|---|---|
| **Query** | Type a question → get an answer with `[[WikiLink]]` citations. Click any link to jump to the article. Reward breakdown (grounding / citation / coverage / fluency) shown inline. |
| **Article** | Browse the full wiki. Collapsible sidebar tree by category. All wikilinks in article bodies are clickable. |
| **Search** | Hybrid TF-IDF + embedding search across the full wiki. |

The top bar shows live stats (article count, trajectory count, LoRA checkpoint, mean reward) and a **⚡ retrain ready** badge when enough new trajectories have accumulated.

---

## How It Works

```
                         ┌─────────────────────────────────────────┐
                         │              WIKI LOOP                  │
                         │                                         │
  raw/papers/  ──┐       │  lore ingest  ──►  chunks.db           │
  raw/articles/ ─┼──────►│  lore absorb  ──►  wiki/*.md           │
  raw/repos/   ──┘       │  (Qwen3-4B)        [[WikiLinks]]        │
                         │                    Obsidian-ready       │
                         └────────────────────────┬────────────────┘
                                                  │
                                           wiki/ search index
                                                  │
                         ┌────────────────────────▼────────────────┐
                         │               RL LOOP                   │
                         │                                         │
  lore query ───────────►│  TF-IDF + embedding retrieval           │
                         │  Qwen3-1.7B + LoRA generates answer     │
                         │  4-signal reward computed               │
                         │  trajectory saved to trajectories.db   │
                         │                                         │
                         │  every 10 trajectories:                 │
                         │    OPD (first 50):  teacher → student   │
                         │    GRPO (after 50): self-improvement    │
                         │    hot-swap LoRA → inference server     │
                         │    divergence guard → auto-rollback     │
                         └─────────────────────────────────────────┘
```

### Reward Function

Every response is scored by four signals, composited into a single scalar reward that drives GRPO:

| Signal | Weight | Mechanism |
|---|---|---|
| **Grounding** | 0.40 | Fraction of response sentences with TF-IDF cosine ≥ 0.25 to any retrieved chunk |
| **Citation** | 0.25 | `[[WikiLink]]` precision — link exists in wiki AND was in the retrieved context |
| **Coverage** | 0.20 | Qwen3-4B judge scores answer completeness 0–4 (offline, cached, not counted in latency) |
| **Fluency** | 0.15 | Unique bigram ratio × hedge factor (penalizes repetition and hallucinated certainty) |

The reward is deliberately designed to be gameable only by being genuinely good: you can't score high on Grounding without staying close to evidence, and you can't score high on Citation without the wiki articles actually existing.

---

## Features

| Feature | Description |
|---|---|
| **Local-first** | All inference runs on your GPU. No OpenAI key, no cloud costs. |
| **Dual-loop learning** | Wiki accumulates externalized knowledge; LoRA bakes it into weights. |
| **SHA-256 dedup** | Ingest the same paper twice — nothing happens. |
| **Hybrid search** | TF-IDF + sentence-embedding retrieval with RRF fusion. |
| **WikiLink repair** | Broken `[[links]]` are auto-snapped to the closest existing article. |
| **Obsidian-compatible** | `wiki/` is a valid Obsidian vault. Graph view, backlinks, everything. |
| **Background training** | GRPO runs in a subprocess; inference never blocks. |
| **Hot-swap adapters** | New LoRA checkpoint replaces the old one without restarting the server. |
| **Divergence guard** | Auto-rollback if reward drops > 1σ below historical baseline. |
| **OPD bootstrap** | First 50 trajectories use teacher distillation (Qwen3-4B → 1.7B) before switching to GRPO. |
| **Florence-2 captions** | Images in `raw/images/` are auto-captioned and embedded in wiki articles. |
| **Marp slides** | `lore render slides <topic>` generates a presentation-ready slide deck. |
| **Health audits** | Find orphaned articles, broken links, stubs, and undiscovered concept connections. |
| **Trajectory export** | Full JSONL export for offline analysis or transfer learning. |

---

## CLI Reference

```bash
# Ingestion
lore ingest <path|url>          # Parse and chunk a paper, article, or repo

# Compilation
lore absorb [--force]           # Compile raw/ → wiki/ via Qwen3-4B
lore rebuild-index              # Refresh TF-IDF and embedding indexes

# Knowledge retrieval
lore search <query>             # Hybrid search, returns ranked articles
lore query "<question>"         # RAG Q&A + captures training trajectory

# Wiki maintenance
lore status                     # Stats: article counts, index age, RL queue
lore health                     # Audit broken links, orphans, stubs
lore cleanup [--dry-run]        # Fix wikilinks, rebuild backlinks
lore reorganize [--apply]       # Detect + fix taxonomy misclassification

# Output generation
lore render report <topic>      # Long-form research report (Markdown)
lore render slides <topic>      # Marp slide deck
lore render charts              # Backlink graph, growth curve, reward history

# RL training management
lore-train status               # Reward stats, checkpoint history, current mode
lore-train train                # Trigger training on accumulated trajectories
lore-train rollback [-n N]      # Roll back N checkpoints (default: 1)
lore-train serve                # Start inference server with hot-swap (port 8765)

# Web UI
lore-web [--port PORT]          # Local web UI at http://localhost:7860
```

---

## Models

All models are loaded from local cache (`/home/ec2-user/SageMaker/hf_cache/`). No internet required after first pull.

| Model | Role | VRAM |
|---|---|---|
| `Qwen/Qwen3-4B` | Wiki compilation, Q&A judge, health checks | ~8 GB (bf16) |
| `Qwen/Qwen3-1.7B` | LoRA training target — the evolving agent | ~4 GB (bf16) |
| `microsoft/Florence-2-base` | Image captioning for `raw/images/` | ~1 GB |

On a 24 GB GPU (A10G, RTX 3090/4090): all three can coexist. On 16 GB: Qwen3-4B and Florence-2 unloaded during training. CPU-only: ingestion and search work fine; training and embedding generation require GPU or MPS.

### Apple Silicon (Mac M1/M2/M3/M4)

Lore runs fully on Mac. Device is detected automatically — no config needed.

```bash
# Same install, same commands
bash scripts/setup.sh && pip install -e .
lore ingest paper.pdf
lore absorb
lore query "..."
lore-train train   # trains on MPS
```

**Memory requirements** (unified memory):

| Setup | Min RAM | Notes |
|---|---|---|
| Ingest + search only | 8 GB | No model loaded |
| Query (inference) | 16 GB | Qwen3-1.7B in float16 ~3.5 GB |
| Absorb (compile) | 16 GB | Qwen3-4B in float16 ~8 GB |
| Training | 16 GB | 1.7B + LoRA + optimizer ~6 GB |
| All at once | 24 GB | M2/M3 Max / M4 Pro recommended |

**Notes:**
- Models load in `float16` on MPS (bfloat16 has incomplete op coverage on MPS as of PyTorch 2.3)
- `bitsandbytes` 4-bit quantization is not used — Lore loads full-precision LoRA, which is compatible with MPS
- Training on MPS is ~3–5× slower than an A10G but works correctly
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` env var can help if you hit memory pressure during training

---

## The Evolving Agent

This is the part that makes Lore different from every other RAG tool.

**The core idea:** every time you ask a question, the answer — good or bad — becomes a training signal. Lore captures the full trajectory: the question, the retrieved context, the generated answer, and a computed reward. After every 10 new trajectories, a training run fires in the background.

**Phase 1 — OPD Bootstrap (trajectories 1–50):** Before the small model (Qwen3-1.7B) has seen enough data to improve via RL, Lore uses "online policy distillation." Qwen3-4B generates a reference answer for each trajectory; Qwen3-1.7B is SFT-trained to match it. This gives the smaller model a strong starting point on your specific research domain before the RL loop kicks in.

**Phase 2 — GRPO (trajectory 51+):** Once the model has a reasonable baseline, Lore switches to Group Relative Policy Optimization. For each query, G=4 responses are sampled from the current policy. All four are scored with the reward function. The reward is normalized within the group (subtract mean, divide by std), and the policy gradient update pushes the model toward higher-reward responses relative to its own rollouts — no separate critic, no reference model needed.

**Hot-swap:** The inference server (FastAPI, port 8765) holds the LoRA adapter in memory. After each training run, the new checkpoint is written to `data/lora_checkpoints/step-NNNNNN/`, and the server swaps adapters without restarting. There is no service interruption.

**Divergence guard:** After each checkpoint swap, Lore computes the mean reward over the last 20 trajectories. If it drops more than 1 standard deviation below the trailing 100-trajectory baseline, the previous checkpoint is automatically restored and a warning is logged. This prevents a bad batch of trajectories from degrading the model.

**What this means in practice:** The model that answers your question after 200 trajectories has been shaped by the specific questions you care about, the specific papers you've read, and the specific ways your wiki is organized. It's not a general assistant that happens to have retrieved some context — it's a model that has internalized your research knowledge.

---

## Directory Layout

```
lore/
├── raw/                    ← Source documents (never edit)
│   ├── papers/             ← PDFs, arXiv exports
│   ├── articles/           ← Web Clipper exports, blog posts
│   ├── repos/              ← README + key files from GitHub repos
│   ├── images/             ← Downloaded figures (Florence-2 captioned)
│   └── notes/              ← Freeform notes, meeting notes
│
├── wiki/                   ← LLM-compiled wiki (Obsidian vault root)
│   ├── concepts/           ← Core ideas: quantization, attention, LoRA...
│   ├── techniques/         ← Methods: GPTQ, AWQ, QAT, SmoothQuant...
│   ├── papers/             ← One article per paper
│   ├── models/             ← Architecture summaries
│   ├── datasets/           ← Dataset provenance and stats
│   ├── benchmarks/         ← Eval benchmark details
│   ├── people/             ← Researcher profiles
│   └── meta/               ← Reading lists, open questions, agendas
│
├── outputs/                ← Generated artifacts
│   ├── queries/            ← Q&A reports: [timestamp]-[slug].md
│   ├── slides/             ← Marp presentations
│   └── charts/             ← matplotlib figures
│
├── src/lore/               ← Python package
│   ├── ingest/             ← parsers · chunker · pipeline (SHA-256 dedup)
│   ├── compile/            ← compiler · taxonomy · linker · incremental
│   ├── index/              ← TF-IDF + embedding store · hybrid RRF search
│   ├── query/              ← RAG agent · citation validation
│   ├── render/             ← reports · Marp slides · matplotlib charts
│   ├── health/             ← contradiction/orphan/stub detection
│   └── evolve/             ← trajectory · reward · GRPO · hot-swap server
│
├── data/                   ← Persistent state (gitignored)
│   ├── chunks.db           ← Ingested chunks (sqlitedict)
│   ├── fingerprints.db     ← SHA-256 dedup registry
│   ├── tfidf_index.pkl     ← TF-IDF vectorizer + sparse matrix
│   ├── embeddings.db       ← Article embeddings (numpy bytes)
│   ├── trajectories.db     ← Q&A trajectories for RL training
│   └── lora_checkpoints/   ← Saved LoRA adapters (step-NNNNNN/)
│
└── scripts/
    ├── setup.sh            ← Install all dependencies, verify GPU + models
    └── bootstrap_wiki.py   ← First-time full compile from raw/
```

---

## Roadmap

| Status | Item |
|---|---|
| Done | Wiki loop: ingest → absorb → Obsidian-compatible markdown |
| Done | Hybrid TF-IDF + embedding search with RRF fusion |
| Done | 4-signal reward function (grounding, citation, coverage, fluency) |
| Done | OPD bootstrap + GRPO training pipeline |
| Done | Hot-swap LoRA inference server with divergence guard |
| Done | Florence-2 image captioning |
| In progress | Multi-document contradiction detection in `lore health` |
| In progress | `lore render charts` — reward history and wiki growth visualizations |
| Done | Web UI — research terminal UI with Query / Article / Search panels (`lore-web`) |
| Planned | Obsidian plugin for inline `lore query` inside notes |
| Planned | Multi-GPU training support (FSDP for the GRPO step) |
| Planned | Cross-wiki federation — share article indexes without sharing raw sources |
| Planned | RLHF preference collection ("was this answer helpful?") |
| Planned | Export trained LoRA for upload to HuggingFace |

---

## Obsidian Setup

1. Open Obsidian → **File → Open Vault** → select `wiki/`
2. `_index.md` and `_summaries.md` are auto-generated — exclude them from search in `.obsidian/` settings
3. Install **Advanced Slides** or **Marp for Obsidian** to preview `outputs/slides/` decks inside Obsidian
4. Use **Obsidian Web Clipper** to save articles directly to `raw/articles/`, then `lore ingest raw/articles/<file>.md && lore absorb`

---

## Contributing

Lore is opinionated about the two-loop architecture but there's a lot of room:

- **New reward signals** — add a new scoring function in `src/lore/evolve/reward.py` and wire it into `RewardConfig`
- **New ingest parsers** — add a `Parser` subclass in `src/lore/ingest/parsers/` for a new file type
- **Better GRPO** — the training loop is in `src/lore/evolve/trainer.py` — KL penalty tuning, different group sizes, etc.
- **New render targets** — `src/lore/render/` for new output formats

Please read `demos/architecture.md` before making changes to the evolve/ module — the OPD/GRPO transition and divergence guard interact in non-obvious ways.

```bash
git clone https://github.com/you/lore
cd lore
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE).

Built on [Qwen3](https://huggingface.co/Qwen), [Florence-2](https://huggingface.co/microsoft/Florence-2-base), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), and [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
