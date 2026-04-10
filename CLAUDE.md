# Lore — Personal Knowledge Wiki Schema

You are maintaining a personal knowledge wiki. This document is your operating manual. You and the user co-evolve this schema over time — if conventions aren't working, suggest changes.

## The Three Layers

1. **`raw/`** — Immutable source documents. You read from here, never write.
2. **`wiki/`** — LLM-maintained wiki (Obsidian vault). You own this entirely. You create pages, update them, maintain cross-references, flag contradictions.
3. **This file** — The schema. It tells you how the wiki is structured, what conventions to follow, and what workflows to run.

## Directory Structure

```
raw/                    ← source documents (NEVER edit)
  papers/               ← academic papers (PDFs, arXiv, OpenReview)
  articles/             ← web articles, blog posts, Obsidian Web Clipper exports
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
- If you reference a concept that doesn't have an article yet, **still use `[[WikiLink]]`** — it becomes a stub (ghost node in Obsidian's graph). Stubs are useful: they show knowledge gaps. During lint, fill stubs by synthesizing from existing wiki content, or suggest sources the user should look for.
- **One canonical name per concept.** Don't create both `[[AWQ]]` and `[[AWQ Paper]]` as separate links. Pick one name and use it everywhere. A paper summary and its technique can be the same article.

## Categories

Categories are not fixed. Start with a few broad ones and let more emerge as the wiki grows. Create a new subdirectory in `wiki/` when an existing category is getting too broad or when a cluster of articles clearly belongs together.

Starting categories (adapt to your domain):

| Category | What goes here |
|---|---|
| `concepts/` | Foundational ideas, theory, overviews |
| `papers/` | One summary page per source document |
| `meta/` | Reading lists, open questions, overview, evolving synthesis |

Add more as needed — `techniques/`, `people/`, `models/`, `datasets/`, `benchmarks/`, or whatever makes sense for your domain. If you're reading a book, categories might be `characters/`, `themes/`, `plot/`. If it's business intelligence, maybe `companies/`, `markets/`, `products/`. Let the content dictate the structure.

When categories feel wrong, use `uv run lore reorganize` to propose reclassification.

## Workflows

### Ingest

When the user adds a new source (drops a file in `raw/`, gives you a file path, or gives you a URL):

1. **Get the source into `raw/`**:
   - If the user dropped a file into `raw/` already → just read it
   - If the user gives a URL → shell out to `uv run lore ingest <url>`. For arXiv links it downloads the actual PDF to `raw/papers/` and extracts text via pdfminer. For web articles it fetches HTML, converts to markdown, and saves to `raw/articles/`. Then read the saved/extracted content.
   - If it's a PDF → shell out to `uv run lore ingest <path>` to extract text, then read the extracted markdown.
2. **Discuss and STOP**: share key takeaways with the user, then **wait for their response**. The user may want to guide what to emphasize, what to skip, what to compare, or what categories matter. Do NOT proceed to writing articles until the user confirms or responds. This is a conversation, not a pipeline.
3. **Write a summary page**: create a page in `papers/` (or the appropriate category) summarizing the source itself — its key findings, methods, and claims
4. **Update related articles**: for each key concept, technique, or person in the source:
   - If an article exists → read it, update it with new information, note contradictions
   - If no article exists → create one following the article format above
   - A single source typically touches 5–15 wiki articles
5. **Cross-reference**: ensure all new articles link to related existing articles, and update existing articles to link back
6. **Note important references**: the source likely references other papers, methods, or people. Create `[[WikiLink]]` stubs for the important ones — they'll show as knowledge gaps in the graph. Suggest which referenced sources are worth ingesting next (e.g. "This paper builds heavily on GPTQ — want me to ingest that next?").
7. **Update `_index.md`**: add entries for any new articles with one-line summaries
8. **Update `_log.md`**: append an entry: `## [YYYY-MM-DD] ingest | Source Title`

**Local mode alternative:** `lore ingest` + `lore absorb` can run the entire wiki loop using the local LLM (Qwen3-4B) without an agent harness. This is useful for bulk-importing many sources, for running on machines without cloud access, or when the user prefers the CLI workflow. Both modes produce the same wiki format and are interchangeable.

### Query

When the user asks a question about the wiki:

1. **Read `_index.md`**: scan the article catalog to identify relevant pages by title and summary — this is your primary navigation tool, not embedding search
2. **Read the articles**: open the relevant wiki pages and read them
3. **Answer**: synthesize with `[[WikiLink]]` citations to the articles you used
4. **File back automatically**: if your answer contains a comparison, synthesis, new connection, or analysis — write it into the wiki without asking. The user's explorations should compound, not disappear into chat history. Rules:
   - Comparison or analysis → create a new article (e.g. `concepts/gptq-vs-awq.md`)
   - New fact or insight about an existing topic → update that article's Key Claims
   - Research gap or open question → add to `meta/open-questions.md`
   - Trivial or conversational answers (yes/no, clarifications, small talk) → don't write anything
   - When in doubt, write it. It's cheap to delete, expensive to lose.
5. **Capture trajectory** (optional, for RL loop): shell out to `uv run lore query "<question>"` — this runs the *local* model (Qwen3-1.7B) on the same question, computes a reward, and saves a training trajectory. It's not for answering the user's question (you already did that); it's for generating RL training data.

At small-to-moderate scale (~hundreds of articles), the index file is all you need. Only shell out to `uv run lore search "<query>"` if the wiki has grown large enough that scanning the index is insufficient.

### Lint / Health

Periodically (or when the user asks), audit the wiki:

1. Shell out to `uv run lore health` for an automated scan
2. Review the report at `wiki/_meta/health_report.md`
3. Fix what you find:
   - **Broken links**: snap to closest match or create the missing article
   - **Orphan articles**: add inbound links from related articles
   - **Stubs**: concepts referenced but lacking articles — fill them by synthesizing from what the wiki already knows. If there's not enough info, suggest sources the user should look for.
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

## Current Context

When discussing a topic, if it's natural and relevant, mention how it relates to what's current — trending repos, recent papers, what the community has adopted, whether the technique is still relevant or has been superseded. Don't force this into every conversation, but when the user is exploring a concept, grounding it in "where things stand today" is valuable. A web search can fill these gaps.

For example, if the user ingests a 2022 GPTQ paper, it's natural to note "GPTQ is still widely used via AutoGPTQ and llama.cpp, but GGUF quantization has largely replaced it for local inference." File these observations into the wiki article when substantive.

## Diagrams

When drawing diagrams (architecture, flowcharts, comparisons, timelines), use Mermaid syntax inside a ` ```mermaid ` code block. Obsidian renders these natively — no plugins needed. When a diagram is part of a substantive answer, file it into the wiki article along with the text.

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
