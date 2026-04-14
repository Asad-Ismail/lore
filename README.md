<div align="center">

```
                                       | |    ___  _ __ ___
                                       | |   / _ \| '__/ _ \
                                       | |__| (_) | | |  __/
                                       |_____\___/|_|  \___|
```

</div>

<p align="center"><strong>Feed it anything. Ask questions. A local model learns your curiosity and tells you what to explore next.</strong></p>

<p align="center"><em>Agent-maintained wiki with proactive follow-up questions. Works with Claude Code, Cursor, Codex, and any MCP-compatible client.</em></p>

https://github.com/user-attachments/assets/63fdbac3-75c5-4947-94b0-27c3ce5fb6ab

Not RAG. Not a chatbot wrapper. Your agent reads sources, writes interlinked wiki articles, and maintains cross-references — while a local LLM trains on your questioning patterns and suggests what to explore next. Inspired by [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

---

## Quick Start

```bash
git clone git@github.com:Asad-Ismail/lore.git && cd lore
```

Open in your agent and start talking:

```
"Ingest https://arxiv.org/abs/2306.00978"
"What is the tradeoff between GPTQ and AWQ?"
"Ingest this PDF from my meeting notes"
"Run a health check"
```

Or just talk normally — ask questions, and it builds up your wiki over time.

Open `wiki/` in Obsidian (or any markdown editor). Watch the graph grow.

On a fresh clone, the wiki is empty and there's no trained model. Suggestions appear automatically after a few questions and a curiosity training run (configurable). Everything builds up from use.

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
| `wiki/` | Interlinked markdown wiki | Agent — reads, writes, maintains |
| `CLAUDE.md` | Schema — conventions, workflows | You and agent co-evolve |

**The agent navigates via `_index.md`** — a catalog of every article with one-line summaries. No vector DB, no embeddings. Works surprisingly well at hundreds of articles.

**What it looks like:** `wiki/` is plain markdown with `[[WikiLinks]]`. View it in Obsidian, VS Code, or any markdown editor. Obsidian's graph view is particularly nice for seeing the knowledge map take shape.

---

## The Curiosity Loop

Every question you ask gets recorded as a trace: the question + the wiki state when you asked it. A local LLM (default: Qwen3-1.7B + LoRA) trains on your traces, first by imitating your questions (SFT), then by optimizing a 4-signal reward (GRPO):

| Signal | Weight | What it measures |
|---|---|---|
| Gap-targeting | 0.35 | Does the question target a knowledge gap in the wiki? |
| Style match | 0.25 | Does it sound like how you ask questions? |
| Novelty | 0.25 | Is it something you haven't asked before? |
| Specificity | 0.15 | Is it specific given current wiki depth? |

Those signals are computed deterministically from the candidate question and wiki state,  no LLM-as-judge in the reward.

The daemon starts automatically (via Stop hook), trains when thresholds are crossed, and caches suggestions. You just talk to the agent. Everything else is automated.

**Health checks** audit the wiki for broken links, orphan articles, stubs, and undiscovered connections — then fix what they find.

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

## Agent Setup

```bash
bash scripts/setup.sh   # creates directories, seed files, installs deps
```

| Agent | How to start |
|---|---|
| **Claude Code** | `claude` in repo root — reads `CLAUDE.md` automatically |
| **Cursor** | Open repo — copy `CLAUDE.md` content to `.cursorrules` |
| **Codex** | Open repo — copy `CLAUDE.md` to `AGENTS.md` |

---

<details>
<summary><strong>MCP Server</strong> — connect any MCP-compatible client (Claude Desktop, etc.)</summary>

Lore exposes an MCP server with 14 tools that mirror the full workflow — search, ingest, write articles, health checks, curiosity suggestions. No functionality is lost compared to the agent-based setup.

#### Local (stdio) — Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lore": {
      "command": "uv",
      "args": ["--directory", "/path/to/lore", "run", "lore-mcp", "--transport", "stdio"]
    }
  }
}
```

#### Remote (HTTP) — any MCP client over the network

```bash
# Start the server
uv run lore-mcp --transport http --port 8766

# Expose via Cloudflare tunnel (brew install cloudflared)
cloudflared tunnel --url http://127.0.0.1:8766
# → gives you https://<random>.trycloudflare.com
# MCP endpoint: https://<random>.trycloudflare.com/mcp
```

#### Tools

| Tool | What it does |
|---|---|
| `read_article` | Read any wiki article by path |
| `search_wiki` | TF-IDF search with snippets |
| `search_and_read` | Search + return full article content |
| `write_article` | Create article (auto-updates `_index.md` + `_log.md`) |
| `update_article` | Update existing article (auto-logs) |
| `update_index` | Manual index edits |
| `append_log` | Manual log entries |
| `cleanup_links` | Fix broken wikilinks + rebuild backlinks (auto-logs) |
| `ingest_url` | Download + extract URL to `raw/` (auto-logs) |
| `run_health_check` | Audit wiki for issues (auto-logs) |
| `rebuild_index` | Rebuild TF-IDF search index |
| `generate_suggestions` | Curiosity-driven follow-up questions |
| `capture_trace` | Record question for training |
| `get_status` | System overview |

Server binds to `127.0.0.1` by default. Writes are confined to the Lore repo (`wiki/`, `raw/`, `data/`).

</details>

---

MIT. See [LICENSE](LICENSE).
