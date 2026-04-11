<div align="center">

<pre style="display:inline-block;text-align:left">
 _
| |    ___  _ __ ___
| |   / _ \| '__/ _ \
| |__| (_) | | |  __/
|_____\___/|_|  \___|
</pre>

**Your LLM agent builds a wiki from your sources. A local model learns how you think.**

[Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern + [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) training

</div>

---

Drop a paper. The agent reads it, writes 10 interlinked wiki articles, updates cross-references, flags contradictions. Ask a question. The agent reads the wiki, answers with citations, files the synthesis back as a new article. Every interaction makes the wiki richer.

Meanwhile, a local model watches how you ask questions — what you dig into, what you skip, where you push deeper. After enough traces, it starts suggesting follow-up questions shaped by your research instincts. The model that suggests your 100th question has been trained on the 99 you asked before it.

<!-- demo GIF here -->

---

## Quick Start

```bash
git clone git@github.com:you/lore.git && cd lore
uv sync && bash scripts/setup.sh
```

Open in Cursor, Claude Code, or Codex. Tell the agent:

```
"Ingest https://arxiv.org/abs/2306.00978"
"What is the tradeoff between GPTQ and AWQ?"
"Run a health check"
```

Open `wiki/` in Obsidian. Watch the graph grow.

On a fresh clone, the wiki is empty and there's no trained model. Suggestions appear automatically after ~15 questions and a curiosity training run. Everything builds up from use.

---

## How It Works

```
  ┌───────────────────────────────────────────────────────┐
  │                   WIKI LOOP                           │
  │                                                       │
  │  raw/ ──► agent reads ──► wiki/*.md                   │
  │           [[WikiLinks]], _index.md, _log.md            │
  │           navigates via _index.md (no embeddings)     │
  └────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │                           │
         ▼                           ▼
   lore trace                   Stop hook
   (question trace)             (automatic)
         │                           │
         ▼                           ▼
  ┌───────────────────────────────────────────────────────┐
  │              DAEMON (lore-train serve)                 │
  │        keeps model in memory on :8765                  │
  │                                                       │
  │  Curiosity training (SFT → GRPO)                      │
  │  learns to ask questions like you                     │
  │                                                       │
  │  GET /suggest ──► 2-3 follow-ups (instant)            │
  └───────────────────────────────────────────────────────┘
         │
         ▼
  agent shows suggestions at end of every answer
```

**Three layers:**

| Layer | What | Who |
|---|---|---|
| `raw/` | Source documents (papers, articles, notes) | You — immutable |
| `wiki/` | Interlinked markdown — Obsidian vault | Agent — reads, writes, maintains |
| `CLAUDE.md` | Schema — conventions, workflows | You and agent co-evolve |

**The agent navigates via `_index.md`** — a catalog of every article with one-line summaries. No vector DB, no embeddings. Works surprisingly well at hundreds of articles.

---

## The Curiosity Loop

Every question you ask gets recorded as a trace: the question + the wiki state when you asked it. A local model (Qwen3-1.7B + LoRA) trains on your traces — first by imitating your questions (SFT), then by optimizing a 4-signal reward (GRPO):

| Signal | Weight | What it measures |
|---|---|---|
| Gap-targeting | 0.35 | Does the question target a knowledge gap in the wiki? |
| Style match | 0.25 | Does it sound like how you ask questions? |
| Novelty | 0.25 | Is it something you haven't asked before? |
| Specificity | 0.15 | Is it specific given current wiki depth? |

The daemon starts automatically (via Stop hook), trains when thresholds are crossed, and caches suggestions. You just talk to the agent. Everything else is automated.

---

## CLI

```bash
lore ingest <path|url>       # fetch paper / extract PDF
lore trace "<question>"      # capture question trace (no GPU)
lore search <query>          # TF-IDF search (for large wikis)
lore health                  # audit broken links, orphans, stubs
lore cleanup                 # fix wikilinks, rebuild backlinks
lore status                  # wiki stats

lore-train serve             # start daemon (keeps model in memory)
lore-train curiosity         # train on your questioning patterns
lore-train suggest           # generate follow-up questions
lore-train status            # trace count, checkpoints
lore-train rollback          # roll back checkpoints
```

---

## Setup

```bash
git clone git@github.com:you/lore.git && cd lore
uv sync && bash scripts/setup.sh
```

| Agent | How to start |
|---|---|
| **Cursor** | Open repo — `CLAUDE.md` auto-read |
| **Claude Code** | `claude` in repo root |
| **Codex** | Copy `CLAUDE.md` to `AGENTS.md` |

Open `wiki/` in Obsidian as a vault. Graph view shows the wiki's shape.

---

## Origin

Built on [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern. Extended with [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)-style training and a curiosity loop that learns your questioning patterns. Domain-agnostic — adapt the schema to your needs.

---

MIT. See [LICENSE](LICENSE).
