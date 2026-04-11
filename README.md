<div align="center">

```
                                        _
                                       | |    ___  _ __ ___
                                       | |   / _ \| '__/ _ \
                                       | |__| (_) | | |  __/
                                       |_____\___/|_|  \___|
```

</div>

<p align="center"><strong>Not RAG. Your LLM agent compiles knowledge into a persistent wiki that compounds with every source and every question. A local model learns your research instincts and suggests what to explore next.</strong></p>

<p align="center">Built on <a href="https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f">Karpathy's LLM Wiki</a> + <a href="https://github.com/Gen-Verse/OpenClaw-RL">OpenClaw-RL</a></p>

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
                              raw/ ──> agent reads ──> wiki/*.md ──> Obsidian
                                            |                |
                                            |          agent answers
                                            |          with [[WikiLinks]]
                                            |                |
                                            v                v
                                       lore trace      files answer
                                       (question +     back into wiki
                                        wiki state)         |
                                            |               |
                                            v               v
                                   ┌────────────────────────────────┐
                                   |    daemon (lore-train serve)   |
                                   |    Qwen3-1.7B + LoRA in memory|
                                   |                                |
                                   |    trains on your questions    |
                                   |    SFT --> GRPO                |
                                   |                                |
                                   |    /suggest --> 2-3 follow-ups |
                                   └────────────────────────────────┘
                                            |
                                            v
                                   agent shows suggestions
                                   at end of every answer
                                            |
                                            v
                                         (repeat)
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
