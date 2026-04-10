<div align="center">

<pre style="display:inline-block;text-align:left">
 _
| |    ___  _ __ ___
| |   / _ \| '__/ _ \
| |__| (_) | | |  __/
|_____\___/|_|  \___|
</pre>

**A personal knowledge wiki maintained by your LLM agent. Knowledge compounds with every source you add and every question you ask.**

Based on [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) + [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)-style RL training

</div>

---

## What This Is

Your LLM agent ([Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) builds and maintains a persistent, interlinked wiki from your raw sources, not RAG, but compiled knowledge that compounds over time. Lore extends this with an [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)-style RL loop that trains a local model on your query trajectories via OPD → GRPO.

---

## Quick Start

```bash
git clone git@github.com:you/lore.git && cd lore
uv sync   # requires uv: https://docs.astral.sh/uv/
```

Open the repo in your agent (Cursor, Claude Code, Codex). The agent reads `CLAUDE.md` and knows how to maintain the wiki. Tell it:

```
"Ingest raw/papers/some-paper.pdf"
"Ingest https://arxiv.org/abs/2306.00978"
"What are the key differences between X and Y?"
"Run a health check on the wiki"
```

The agent reads and writes `wiki/` directly, navigating via `_index.md`. Open `wiki/` in Obsidian to browse in real time.

For local-only use (no agent harness, needs GPU):

```bash
uv run lore ingest raw/papers/paper.pdf    # parse source
uv run lore absorb                          # compile into wiki via Qwen3-4B
uv run lore query "your question"           # Q&A + RL trajectory capture
uv run lore-train train                     # fine-tune local model
```

---

## How It Works

### The Three Layers

| Layer | What | Who owns it |
|---|---|---|
| `raw/` | Source documents — papers, articles, repos, notes, images | You (immutable, never edited) |
| `wiki/` | Interlinked markdown articles — Obsidian vault | The agent (reads, writes, maintains) |
| `CLAUDE.md` | Schema — structure, conventions, workflows | You and the agent co-evolve |

### Operations

**Ingest.** Drop a source into `raw/` and tell the agent to process it. It reads the source, discusses key takeaways, writes a summary page, updates related articles across the wiki, maintains cross-references, and logs what happened. A single source typically touches 5–15 wiki pages.

**Query.** Ask a question. The agent reads `_index.md` to find relevant pages, reads them, synthesizes an answer with `[[WikiLink]]` citations. Good answers get filed back into the wiki so your explorations compound.

**Lint.** The agent health-checks the wiki: finds broken links, orphan articles, stubs, undiscovered connections, contradictions. Then suggests new questions to investigate and new sources to look for.

### Navigation

- **`_index.md`** — catalog of every article with one-line summaries, organized by category. The agent reads this first when answering queries. This is the retrieval mechanism — no embeddings needed at moderate scale.
- **`_log.md`** — chronological record of operations. Parseable with `grep "^## \[" wiki/_log.md | tail -5`.

---

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │              WIKI LOOP                   │
                     │          (agent-maintained)              │
                     │                                          │
  raw/papers/  ──┐   │  agent reads source directly             │
  raw/articles/ ─┼──►│  agent writes/updates wiki/*.md          │
  raw/repos/   ──┘   │  agent maintains [[WikiLinks]]           │
                     │  agent updates _index.md and _log.md     │
                     └────────────────┬─────────────────────────┘
                                      │
                        agent navigates wiki via _index.md
                        (no embeddings, no vector search)
                                      │
                                      ▼
                     ┌──────────────────────────────────────────┐
                     │         RL LOOP (optional)               │
                     │   (local model, runs separately)         │
                     │                                          │
  lore query ───────►│  local LLM + LoRA reads wiki     │
  (trajectory        │  generates its own answer                │
   capture)          │  4-signal reward computed                │
                     │  trajectory saved                        │
                     │                                          │
                     │  OPD (first 50) → GRPO (after 50)       │
                     │  divergence guard → auto-rollback        │
                     └──────────────────────────────────────────┘
```

---

## CLI Tools

Helpers the agent shells out to. Install with `uv sync`.

```bash
# Ingestion
lore ingest <path|url>          # Extract text from PDFs/binary formats
lore absorb [--force]           # Batch-compile sources via local LLM

# Search (optional — agent uses _index.md by default)
lore search <query>             # Hybrid search for large wikis
lore query "<question>"         # Local model Q&A + RL trajectory capture

# Maintenance
lore health                     # Audit broken links, orphans, stubs, connections
lore cleanup [--dry-run]        # Fix wikilinks, rebuild backlinks
lore reorganize [--apply]       # Detect taxonomy misclassification
lore status                     # Wiki stats

# Output
lore render report <topic>      # Long-form markdown report
lore render slides <topic>      # Marp slide deck

# RL training
lore-train status               # Reward stats, checkpoint history
lore-train train [--background] # Train on accumulated trajectories
lore-train rollback [-n N]      # Roll back checkpoints
```

---

## RL Training (Extension)

Beyond the base wiki pattern, Lore trains a local model that improves with use. Every `lore query` captures a trajectory scored by 4 signals:

| Signal | Weight | What it measures |
|---|---|---|
| Grounding | 0.40 | Response grounded in retrieved wiki content |
| Citation | 0.25 | `[[WikiLink]]` precision — links are valid and in context |
| Coverage | 0.20 | LLM judge scores completeness 0–4 |
| Fluency | 0.15 | Non-degenerate, non-repetitive output |

**Phase 1 (trajectories 1–50):** OPD — Qwen3-4B generates reference answers, Qwen3-1.7B is SFT-trained to match.
**Phase 2 (trajectory 51+):** GRPO — G=4 responses sampled, scored, policy gradient toward higher reward.
**Divergence guard:** auto-rollback if reward drops > 1σ below baseline.

Local models (used by CLI tools, not by the agent harness):

| Model | Role | VRAM |
|---|---|---|
| `Qwen/Qwen3-4B` | Teacher, coverage judge, batch compilation | ~8 GB |
| `Qwen/Qwen3-1.7B` | LoRA training target | ~4 GB |
| `microsoft/Florence-2-base` | Image captioning | ~1 GB |

---

## Directory Layout

```
lore/
├── raw/                    ← Source documents (never edit)
│   ├── papers/
│   ├── articles/
│   ├── repos/
│   ├── images/
│   └── notes/
├── wiki/                   ← Agent-maintained wiki (Obsidian vault)
│   ├── concepts/           ← Core ideas and overviews
│   ├── techniques/         ← Specific methods
│   ├── papers/             ← Summary page per source
│   ├── models/             ← System/architecture summaries
│   ├── datasets/           ← Dataset provenance
│   ├── benchmarks/         ← Evaluation benchmarks
│   ├── people/             ← Key people
│   ├── meta/               ← Reading lists, open questions, synthesis
│   ├── _index.md           ← Article catalog
│   └── _log.md             ← Operation log
├── outputs/                ← Reports, slides, charts
├── src/lore/               ← CLI tools (Python)
├── data/                   ← Persistent state (gitignored)
├── CLAUDE.md               ← Wiki schema
└── pyproject.toml
```

---

## Obsidian Setup

1. **File → Open Vault** → select `wiki/`
2. **Graph view** shows the wiki's shape: hubs, orphans, clusters
3. Install **Obsidian Web Clipper** to save articles to `raw/articles/`
4. Install **Dataview** for frontmatter queries (tags, dates, source counts)
5. Set attachment folder to `raw/images/` in Settings → Files and Links
6. The wiki is a git repo — version history and branching for free

---

## Agent Harnesses

| Setup | How to start |
|---|---|
| **Cursor** | Open repo — `CLAUDE.md` auto-read as workspace rules |
| **Claude Code** | Run `claude` in the repo root |
| **OpenAI Codex** | Copy `CLAUDE.md` to `AGENTS.md` |
| **Local only** | `uv sync` + CLI tools. Needs GPU or MPS. |

---

## Why This Works

The tedious part of a knowledge base is the bookkeeping — cross-references, summaries, contradictions, consistency. Humans abandon wikis because maintenance burden grows faster than value. LLMs don't get bored, don't forget a cross-reference, and can touch 15 files in one pass.

Your job: curate sources, direct analysis, ask good questions. The agent's job: everything else.

---

## Origin

Lore instantiates the [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern by [Andrej Karpathy](https://github.com/karpathy) — a design for personal knowledge bases maintained by LLM agents. Lore adds an [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)-style RL training loop (OPD → GRPO) that trains a local model on your query trajectories. The pattern is domain-agnostic — adapt the schema and categories to your needs.

---

## License

MIT. See [LICENSE](LICENSE).
