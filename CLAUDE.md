# Lore — Personal Knowledge Wiki Schema

You are maintaining a personal knowledge wiki. This document is your operating manual. You and the user co-evolve this schema over time — if conventions aren't working, suggest changes.

## The Three Layers

1. **`raw/`** — Immutable source documents. You read from here, never write.
2. **`wiki/`** — LLM-maintained wiki (Obsidian vault). You own this entirely. You create pages, update them, maintain cross-references, flag contradictions.
3. **This file** — The schema. It tells you how the wiki is structured, what conventions to follow, and what workflows to run.

## Directory Structure

```
raw/                    ← source documents (NEVER edit)
  papers/               ← PDFs, arXiv markdown
  articles/             ← Obsidian Web Clipper exports, blog posts
  repos/                ← README + key files from GitHub repos
  images/               ← downloaded figures
  notes/                ← freeform notes, meeting notes

wiki/                   ← you maintain this (Obsidian vault root)
  concepts/             ← core ideas and overviews
  techniques/           ← specific methods and algorithms
  papers/               ← one summary page per paper/source
  models/               ← architecture or system summaries
  datasets/             ← dataset provenance and stats
  benchmarks/           ← evaluation benchmarks
  people/               ← profiles of key people
  meta/                 ← reading lists, open questions, research agendas, overview/synthesis
  _index.md             ← auto-maintained catalog of all articles
  _log.md               ← chronological record of operations

outputs/                ← generated artifacts (reports, slides, charts)
data/                   ← persistent state (gitignored): SQLite DBs, indexes, LoRA checkpoints
```

## Article Format

Every wiki article must follow this structure:

```markdown
---
title: Article Title
category: techniques
created: 2026-04-10T12:00:00+00:00
updated: 2026-04-10T12:00:00+00:00
sources:
  - raw/papers/awq-paper.md
  - raw/articles/quantization-survey.md
---

# Article Title

2-sentence definition/summary.

## Context

Where this fits in the broader landscape.

## Key Claims

Numbered, citable claims from sources. Each claim should reference
which source it comes from.

## Connections

How this relates to other wiki articles. Use [[WikiLink]] syntax.

## Sources

- [[awq-paper]] — Section 3, activation-aware scaling
- [[quantization-survey]] — Table 2, INT4 comparison

## Referenced By

<!-- auto-maintained by cleanup — do not edit -->
- [[Post Training Quantization]]
- [[Activation Aware Quantization]]
```

## WikiLink Conventions

- Use `[[Article Title]]` to link to other wiki articles
- The link target must match an existing article title (case-insensitive, hyphens = spaces)
- If you reference a concept that doesn't have an article yet, still use `[[WikiLink]]` — it becomes a stub candidate that `lore health` will detect
- Prefer linking generously: every concept, technique, paper, or person mentioned should be a link if it has (or should have) its own article

## Category Rules

| Category | What goes here | Examples |
|---|---|---|
| `concepts/` | Foundational ideas, theory, overviews | Any core concept in the domain |
| `techniques/` | Specific methods and algorithms | Named methods, processes, tools |
| `papers/` | One summary page per paper or source document | Summary of key findings + citations back to raw/ |
| `models/` | Architecture or system summaries | Named systems, architectures, products |
| `datasets/` | Dataset provenance | Data sources with stats and provenance |
| `benchmarks/` | Evaluation benchmarks | Named benchmarks, metrics |
| `people/` | Profiles of key people | Researchers, authors, leaders |
| `meta/` | Reading lists, open questions, overview, synthesis | Evolving thesis, research agendas, the "big picture" page |

## Workflows

### Ingest

When the user adds a new source (drops a file in `raw/` or gives you a URL):

1. **Read the source yourself**: open the file in `raw/` and read it. For PDFs, shell out to `uv run lore ingest <path>` to extract text first, then read the extracted markdown.
2. **Discuss**: briefly share key takeaways with the user
3. **Write a summary page**: create a page in `papers/` (or the appropriate category) summarizing the source itself — its key findings, methods, and claims
4. **Update related articles**: for each key concept, technique, or person in the source:
   - If an article exists → read it, update it with new information, note contradictions
   - If no article exists → create one following the article format above
   - A single source typically touches 5–15 wiki articles
5. **Cross-reference**: ensure all new articles link to related existing articles, and update existing articles to link back
6. **Update `_index.md`**: add entries for any new articles with one-line summaries
7. **Update `_log.md`**: append an entry: `## [YYYY-MM-DD] ingest | Source Title`

**Local mode alternative:** `lore ingest` + `lore absorb` can run the entire wiki loop using the local LLM (Qwen3-4B) without an agent harness. This is useful for bulk-importing many sources, for running on machines without cloud access, or when the user prefers the CLI workflow. Both modes produce the same wiki format and are interchangeable.

### Query

When the user asks a question about the wiki:

1. **Read `_index.md`**: scan the article catalog to identify relevant pages by title and summary — this is your primary navigation tool, not embedding search
2. **Read the articles**: open the relevant wiki pages and read them
3. **Answer**: synthesize with `[[WikiLink]]` citations to the articles you used
4. **File back**: if the answer contains a novel synthesis, comparison, or connection — write it back into the wiki. This is critical — explorations should compound, not disappear into chat history. Where to put it:
   - A comparison → create a new article in the relevant category (e.g. `concepts/gptq-vs-awq.md`)
   - New insight about an existing topic → update that topic's article, add to Key Claims
   - A research question or gap → add to `meta/open-questions.md`
   - Answers can take different forms: a markdown page, a comparison table, a slide deck (`lore render slides`), a chart. Choose what fits.
5. **Capture trajectory** (optional, for RL loop): shell out to `uv run lore query "<question>"` — this runs the *local* model (Qwen3-1.7B) on the same question, computes a reward, and saves a training trajectory. It's not for answering the user's question (you already did that); it's for generating RL training data.

At small-to-moderate scale (~hundreds of articles), the index file is all you need. Only shell out to `uv run lore search "<query>"` if the wiki has grown large enough that scanning the index is insufficient.

### Lint / Health

Periodically (or when the user asks), audit the wiki:

1. Shell out to `uv run lore health` for an automated scan
2. Review the report at `wiki/_meta/health_report.md`
3. Fix what you find:
   - **Broken links**: snap to closest match or create the missing article
   - **Orphan articles**: add inbound links from related articles
   - **Stubs**: concepts referenced but lacking articles — create them
   - **Undiscovered connections**: article pairs with high similarity but no explicit link — add cross-references
   - **Contradictions**: when two articles make conflicting claims, flag them in both articles under a `## Contradictions` section with the specific claims and sources
   - **Stale claims**: when newer sources supersede older ones, update the article and note what changed
4. **Suggest**: after fixing, suggest new questions to investigate and new sources to look for based on gaps you found. The wiki should grow proactively, not just reactively.
5. Shell out to `uv run lore cleanup` to auto-fix wikilinks and rebuild backlinks
6. Update `_log.md`: `## [YYYY-MM-DD] lint | Fixed N issues`

### Reorganize

When the wiki has grown and categories feel wrong:

1. Shell out to `uv run lore reorganize` (dry-run by default) to see proposals
2. Review the suggestions with the user
3. Apply with `uv run lore reorganize --apply` if agreed

## Special Files

### `_index.md`

Content-oriented catalog. Every article listed with a link and one-line summary, organized by category. You update this on every ingest. **This is your primary retrieval mechanism.** When answering a query, read the index first to find relevant pages, then drill into them. This avoids the need for embedding infrastructure — at moderate scale (~hundreds of pages) it works surprisingly well.

Format:
```markdown
# Wiki Index

## Concepts
- [[Quantization]] — Reducing numerical precision of model weights/activations (12 sources)
- [[KV Cache]] — Key-value cache for autoregressive inference speedup (3 sources)

## Techniques
- [[GPTQ]] — Second-order weight quantization using Hessian information (5 sources, 2023-03)
...
```

Optionally include metadata like source count or date — this helps you (and the user) gauge article maturity at a glance.

### `_log.md`

Chronological, append-only. Each entry starts with a consistent prefix so it's parseable with grep.

Format:
```markdown
# Wiki Log

## [2026-04-10] ingest | AWQ: Activation-aware Weight Quantization
- Created: [[AWQ]], [[Activation Aware Quantization]]
- Updated: [[Post Training Quantization]], [[Quantization Benchmarks]]
- 12 chunks processed

## [2026-04-10] query | "What is the tradeoff between GPTQ and AWQ?"
- Retrieved 8 articles, synthesized comparison
- Filed as: outputs/queries/20260410-gptq-vs-awq.md

## [2026-04-11] lint | Health audit
- Fixed 3 broken links, created 2 stub articles
```

## CLI Tools

These are helpers you shell out to. Install with `uv sync`.

| Command | When to use it |
|---|---|
| `uv run lore ingest <path\|url>` | Extract text from PDFs and other binary formats the agent can't read directly |
| `uv run lore absorb` | Batch-compile unprocessed sources into wiki articles using local LLM (for bulk import without agent supervision) |
| `uv run lore search "<query>"` | Find relevant articles when wiki is too large for `_index.md` scanning |
| `uv run lore query "<question>"` | Run the local model on a question and capture an RL training trajectory (not for answering the user — you do that) |
| `uv run lore rebuild-index` | Refresh the search index (only needed if you use `lore search` or `lore query`) |
| `uv run lore health` | Automated scan for broken links, orphans, stubs, connections |
| `uv run lore cleanup` | Auto-fix broken wikilinks, rebuild backlink footers |
| `uv run lore status` | Wiki stats: article counts, index age, trajectory queue |
| `uv run lore reorganize` | Detect taxonomy misclassification |
| `uv run lore render report <topic>` | Generate a long-form markdown research report |
| `uv run lore render slides <topic>` | Generate a Marp slide deck |
| `uv run lore-train status` | Check RL training status, reward stats, checkpoint history |
| `uv run lore-train train` | Trigger LoRA training on accumulated trajectories |
| `uv run lore-train rollback` | Roll back to a previous LoRA checkpoint |

## RL Training Extension

Beyond the base wiki pattern, Lore trains a local model that improves with use.

**How it works:**
- Every `lore query` captures a trajectory: question, retrieved context, response, reward
- The reward is a composite of 4 signals: grounding (0.40), citation precision (0.25), coverage (0.20), fluency (0.15)
- After 10 new trajectories, Lore suggests retraining — run `uv run lore-train train`
- First 50 trajectories use OPD (teacher distillation: Qwen3-4B → 1.7B)
- After 50: GRPO (self-improvement via group-relative policy optimization)
- Divergence guard auto-rolls back if reward degrades

**Models (local, loaded from HF cache — used by CLI tools, not by you):**
- `Qwen/Qwen3-4B` — teacher model for OPD distillation and coverage judge; also used by `lore absorb` for batch compilation
- `Qwen/Qwen3-1.7B` — LoRA training target, the evolving local model used by `lore query`
- `microsoft/Florence-2-base` — image captioning for `raw/images/`

You (the agent harness) do wiki maintenance: reading sources, writing articles, maintaining cross-references. The local models handle trajectory-scored Q&A and improve over time via RL. Both loops compound knowledge — one as markdown, one as weights.

## Image Handling

Sources in `raw/` may reference images (figures, diagrams, charts). When ingesting a source with images:

1. If images are already downloaded to `raw/images/`, reference them from wiki articles as `![[filename.png]]`
2. Read the text content first, then view referenced images separately for additional context
3. When writing wiki articles, describe what figures show in text — don't rely on the image alone being present

Tip: In Obsidian, set attachment folder to `raw/images/` and use the "Download attachments" command to pull images locally from clipped articles.

## Git

The wiki is just a git repo of markdown files. You get version history, branching, and collaboration for free. Commit after significant ingest or lint sessions so the user has a checkpoint to roll back to.

## Important Rules

- `raw/` is source-of-truth. Never modify files there.
- `wiki/` is yours to maintain. Write freely, but follow the format.
- `data/` is gitignored. Contains SQLite DBs, search indexes, LoRA checkpoints.
- Link generously. The value of the wiki is in its connections.
- When new information contradicts existing articles, update both — don't silently overwrite.
- Every ingest should update `_index.md` and `_log.md`.
- File your query answers back into the wiki when they contain novel synthesis.
