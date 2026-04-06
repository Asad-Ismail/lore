---
name: lore
description: Build and query a personal ML research knowledge base. Compiles raw sources (papers, articles, repos) into a structured markdown wiki, and captures every Q&A as a training trajectory for an evolving local LoRA model.
---

You are managing a personal ML/quantization research knowledge wiki at:
`/home/ec2-user/SageMaker/personal-knowledge/`

Read CLAUDE.md at the repo root before proceeding — it has the full architecture overview.

The wiki is organized as:
- `raw/` — source documents (never modify)
- `wiki/` — compiled wiki articles (Obsidian vault)
- `outputs/` — generated reports, slides, charts
- `data/` — search indexes, LoRA checkpoints, trajectory DB

---

## /lore ingest <source>

**Purpose**: Add a new source document to `raw/` and prepare it for compilation.

**Source types handled**:
- Local file path (`.pdf`, `.md`, `.txt`, `.csv`, image)
- URL — fetch and convert to markdown
- Directory path — recursively ingest all supported files

**Steps**:
1. If URL: use the WebFetch tool to retrieve the content as markdown, save to `raw/articles/<slug>.md`
2. If local file: copy/symlink to appropriate `raw/<type>/` subdirectory
3. Write a `.meta.json` sidecar with: `{"source": "<origin>", "date": "<ISO date>", "type": "<type>", "tags": []}`
4. Run: `python3 -c "from lore.ingest.pipeline import ingest_file; ingest_file('<path>')"` to fingerprint and chunk
5. Report: number of chunks created, SHA-256 fingerprint, destination path

**After ingest**: Remind the user to run `/lore absorb` to compile the new source into the wiki.

---

## /lore absorb [--force]

**Purpose**: Compile all unprocessed sources from `raw/` into wiki articles.

**Steps**:
1. Run: `lore-absorb` (or `python3 -m lore.cli.wiki_cli absorb`)
2. This will:
   - Read all chunks not yet absorbed (tracked by `absorbed: true` frontmatter flag in entries)
   - Group chunks by topic using Qwen3-4B concept extraction
   - For each new concept: write a new `wiki/<category>/<concept>.md` article
   - For existing articles: append/update with new information
   - Inject `[[WikiLink]]` backlinks into all affected articles
   - Write `data/.absorb_pending` flag to trigger index rebuild
3. Report: N new articles, M updated articles, list of new article titles

**Format of wiki articles** (maintain this structure):
```markdown
---
title: <Concept Name>
category: <one of: concepts/techniques/papers/models/datasets/benchmarks/people/meta>
created: <ISO date>
updated: <ISO date>
sources: [<source1>, <source2>]
absorbed: true
---

# <Concept Name>

<2-sentence definition>

## Context
<Background and motivation>

## Key Claims
- <Specific claim from source> [[RelatedConcept]]
- <Another claim>

## Connections
- [[RelatedArticle1]] — <brief explanation of relationship>
- [[RelatedArticle2]] — <brief explanation>

## Sources
- <source document name>, <section/page if available>
```

**After absorb**: The `post_absorb.sh` hook will auto-rebuild the search index.
If the hook is not active, run: `lore-rebuild-index`

---

## /lore query <question>

**Purpose**: Answer a question using the wiki, and capture the interaction as a training trajectory.

**Steps**:
1. Run: `lore-search "<question>" --top-k 8` to retrieve the most relevant wiki articles
2. Read the full content of the top articles returned
3. Also read `wiki/_summaries.md` to check for any relevant articles the search may have missed
4. Formulate a comprehensive answer:
   - Ground every claim in the retrieved wiki articles
   - Use [[ArticleName]] citations for each source article
   - Flag any aspects of the question not covered by the wiki
5. Write the full answer as a markdown report to `outputs/queries/<timestamp>-<slug>.md`
   Format:
   ```markdown
   # Query: <question>
   Date: <ISO datetime>
   
   ## Answer
   <detailed answer with [[citations]]>
   
   ## Evidence
   | Claim | Source Article | Confidence |
   |---|---|---|
   
   ## Gaps
   <What the wiki doesn't cover about this question>
   
   ## Related Topics
   <Suggestions for further wiki exploration>
   ```
6. Show the answer to the user and mention the saved report path

**Note**: Trajectory capture (for RL training) happens automatically inside `lore-query` — no manual step needed.

**Important**: Always read the actual wiki articles, not just summaries. The quality of your answer directly determines the training reward for the evolving agent.

---

## /lore status

**Purpose**: Show current state of the knowledge base.

Run: `lore-status`

This reports:
- Article counts by category (concepts: N, techniques: N, papers: N, ...)
- Total word count across all wiki articles
- Search index: last built timestamp, article count indexed
- Trajectory DB: total trajectories, untrained count, mean reward (last 50)
- LoRA checkpoint: latest version, training steps, reward improvement
- Raw sources: file count by type, unabsorbed count

---

## /lore rebuild-index

**Purpose**: Rebuild the TF-IDF and embedding search index from scratch.

Run: `lore-rebuild-index`

Use this after:
- Bulk editing wiki articles manually
- Major restructuring or taxonomy changes
- Any time `lore-search` returns stale results

Takes 1-3 minutes for a ~500 article wiki.

---

## /lore health

**Purpose**: Audit the wiki for quality issues and discover new connection candidates.

Run: `lore-health`

This produces `wiki/_meta/health_report.md` containing:
- **Broken links**: `[[WikiLinks]]` pointing to non-existent articles
- **Orphan articles**: Articles with no incoming backlinks (isolated nodes)
- **Stub candidates**: Concepts mentioned in `[[WikiLinks]]` but lacking their own article
- **Contradictions**: Articles with conflicting claims on the same topic
- **Undiscovered connections**: Article pairs with high semantic similarity but no explicit link
- **Stale articles**: Articles whose source material has been updated but article hasn't been recompiled

After reviewing the report, you can:
- Fix broken links manually or re-run `/lore absorb` to regenerate affected articles
- Create stub articles for high-priority missing concepts
- Add `[[WikiLink]]` connections for suggested undiscovered pairs

---

## /lore cleanup

**Purpose**: Fix structural issues in the wiki.

Steps:
1. Read `wiki/_meta/health_report.md` (run `/lore health` first if it's stale)
2. Fix broken `[[WikiLinks]]` — either correct the link target or remove it
3. Merge duplicate articles (same concept, different titles) — keep the more comprehensive one, redirect the other
4. Inject missing backlinks — if article A mentions concept B but doesn't link `[[B]]`, add the link
5. Report: N links fixed, M articles merged, K backlinks added

---

## /lore render report <topic>

**Purpose**: Generate a comprehensive long-form research report on a topic.

Steps:
1. Run several `/lore query` subqueries to gather information from multiple angles:
   - "What is <topic>?"
   - "What are the key techniques in <topic>?"
   - "What are the open problems in <topic>?"
   - "What does the latest research say about <topic>?"
2. Synthesize all query results into a structured report
3. Write to `outputs/reports/<topic>-<date>.md`

Report structure:
```markdown
# Research Report: <Topic>
Generated: <date>

## Executive Summary
<3-4 sentence overview>

## Background
## Core Techniques
## Recent Advances
## Open Problems
## Key Papers
## Researchers to Follow
## Confidence Notes
<Note which sections have strong wiki coverage vs. gaps>
```

---

## /lore render slides <topic>

**Purpose**: Generate a Marp slide deck for a topic.

Steps:
1. Gather topic information via `/lore query` subqueries
2. Structure as Marp markdown (slides separated by `---`)
3. Write to `outputs/slides/<topic>-<date>.md`

Slide structure:
```markdown
---
marp: true
theme: default
paginate: true
---

# <Topic>
### Personal Research Wiki
<date>

---

## Overview
- <key point 1>
- <key point 2>

---
<!-- One slide per major concept, technique, or finding -->
```

---

## /lore reorganize

**Purpose**: Re-examine all wiki articles and suggest or apply taxonomy reclassification.

Steps:
1. Read all article titles and first paragraphs
2. For each article, determine if its current category is the best fit given the full taxonomy:
   `concepts / techniques / papers / models / datasets / benchmarks / people / meta`
3. List reclassification suggestions with reasoning
4. Ask for confirmation before moving any files
5. After moves: update all backlinks pointing to moved articles, rebuild index

---

## /lore breakdown <topic>

**Purpose**: Deep-dive into a topic, map its coverage in the wiki, and generate a research agenda.

Steps:
1. Search wiki for all articles related to `<topic>`
2. Map the topic structure: core concepts, sub-techniques, open questions, key papers
3. Identify gaps: what subtopics have no wiki coverage?
4. Generate `wiki/meta/<topic>-research-agenda.md` with:
   - Current coverage map
   - Priority gaps (high/medium/low)
   - Suggested sources to ingest for each gap
   - Research questions to explore

---

## General Guidelines

- **Never edit `raw/`** — it is source-of-truth; always ingest new content via `/lore ingest`
- **Always use `[[WikiLink]]` syntax** — this powers the backlink graph and Obsidian navigation
- **Keep articles focused** — one concept per article; create linked stub articles for related concepts
- **Cite everything** — vague claims without source attribution hurt the RL reward signal
- **Run `/lore rebuild-index` after any manual edits** — keeps search fresh
- The wiki is the domain of the LLM; the user rarely needs to edit articles directly
