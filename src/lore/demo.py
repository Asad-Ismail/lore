"""Seed a reproducible demo workspace for fresh Lore clones."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lore.config import DATA_DIR, OUTPUTS_DIR, RAW_DIR, WIKI_DIR
from lore.evolve.curiosity import build_wiki_state_summary, generate_suggestions_with_mode
from lore.evolve.trajectory import capture_question_trace
from lore.ingest.pipeline import IngestedSource, ingest_file_result, ingest_url_result
from lore.index.store import rebuild_index
from lore.linker import rebuild_all_backlinks, snap_wikilinks

DEMO_TIMESTAMP = "2026-04-14T00:00:00+00:00"

DEMO_SOURCES: dict[Path, str] = {
    Path("articles/llm-wiki.md"): """# LLM Wiki Notes

An LLM wiki turns repeated conversations into durable pages instead of disposable chat history.
The agent writes summaries, comparisons, and open questions back into markdown so the graph gets
better every time you use it.

Practical advantages:
- retrieval starts from an explicit index and cross-linked pages
- the user can inspect and edit everything in plain markdown
- the agent compounds prior questions instead of forgetting them
""",
    Path("articles/model-context-protocol.md"): """# Model Context Protocol Notes

Model Context Protocol gives a client a standard way to call tools exposed by a local or remote
server. For Lore that means search, read, write, cleanup, health checks, and suggestion generation
can be used from any MCP-compatible client without re-implementing the workflow.

The important product property is that the same wiki can serve Claude Code, Codex, Cursor, or a
desktop app through one interface.
""",
    Path("articles/active-memory.md"): """# Active Memory Notes

Active memory is the idea that an agent should deliberately hydrate itself with the right prior
state before answering. Instead of relying on one giant prompt, the agent fetches compact memory,
recent changes, and the exact pages that matter for the current task.

A wiki is a good substrate for active memory because the structure is explicit and user-editable.
""",
    Path("notes/question-trace-principles.md"): """# Question Trace Principles

Question traces record what the user asked together with the state of the knowledge base at the
time of the question. They are useful because follow-up suggestions should match the user's style,
not just the corpus.

Signals worth optimizing:
- target gaps in coverage
- match the user's phrasing and level of specificity
- avoid repeating questions that were already answered
""",
}

DEMO_ARTICLES: dict[Path, str] = {
    Path("concepts/LLM Wiki.md"): f"""---
title: LLM Wiki
category: concepts
created: {DEMO_TIMESTAMP}
updated: {DEMO_TIMESTAMP}
sources:
  - raw/articles/llm-wiki.md
  - raw/notes/question-trace-principles.md
---

# LLM Wiki

An agent-maintained wiki turns repeated chats into durable knowledge. Instead of answering from
scratch each time, the agent writes reusable pages, links them, and keeps a graph you can inspect.

## Context

An LLM wiki sits between ad hoc chat history and heavyweight retrieval stacks. It works best when
the system keeps a clean index, writes summaries back into markdown, and reuses prior questions to
steer what to explore next.

## Key Claims

1. Durable markdown pages make knowledge compound across conversations instead of disappearing into
   chat logs.
2. A curated index plus explicit links often gets you most of the retrieval value before you need
   embeddings or vector infrastructure.
3. Question traces are part of the memory system because they capture what the user actually cares
   to ask next.

## Connections

- [[Model Context Protocol]] exposes the same wiki workflow to multiple clients.
- [[Curiosity Training]] learns from repeated questions and proposes what to explore next.

## Sources

- raw/articles/llm-wiki.md — framing for turning chat into a durable wiki
- raw/notes/question-trace-principles.md — why question traces should shape the system

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
""",
    Path("concepts/Model Context Protocol.md"): f"""---
title: Model Context Protocol
category: concepts
created: {DEMO_TIMESTAMP}
updated: {DEMO_TIMESTAMP}
sources:
  - raw/articles/model-context-protocol.md
---

# Model Context Protocol

Model Context Protocol standardizes how a client calls external tools. In Lore it lets any
MCP-compatible agent search the wiki, read pages, write updates, run health checks, and ask for
follow-up suggestions through one interface.

## Context

The protocol matters because the knowledge base should outlive any single chat client. When the
tool surface is stable, the same personal wiki can be used from Codex, Claude Code, Cursor, or an
MCP desktop app.

## Key Claims

1. MCP turns the wiki into a reusable tool server instead of a client-specific prompt hack.
2. A stable tool interface matters more for long-term leverage than a one-off chat integration.
3. Tool calls are most valuable when they operate on durable state like wiki pages, traces, and
   health reports.

## Connections

- [[Active Memory]] is stronger when clients can fetch the exact pages they need.
- [[Curiosity Training]] can be surfaced through the same tool interface as search and write.

## Sources

- raw/articles/model-context-protocol.md — why a shared tool protocol matters for Lore

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
""",
    Path("concepts/Active Memory.md"): f"""---
title: Active Memory
category: concepts
created: {DEMO_TIMESTAMP}
updated: {DEMO_TIMESTAMP}
sources:
  - raw/articles/active-memory.md
---

# Active Memory

Active memory is the practice of hydrating an agent with the smallest useful slice of prior state
before it answers. The agent should load the right pages, traces, and recent updates instead of
hoping everything relevant is already in the prompt.

## Context

Lore makes active memory concrete because the memory is visible and editable. A user can inspect the
pages that shaped an answer, fix them, and watch the next answer improve.

## Key Claims

1. Active memory is stronger when the memory substrate is structured markdown instead of opaque chat
   history.
2. The system should reload just enough prior state to answer well, not dump the entire corpus into
   every turn.
3. Good active memory needs explicit retrieval surfaces like search, read, and status tools.

## Connections

- [[Model Context Protocol]] lets clients fetch the right wiki state on demand.
- [[Curiosity Training]] benefits when the agent can see recent traces and unresolved gaps.

## Sources

- raw/articles/active-memory.md — notes on memory hydration for agent workflows

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
""",
    Path("concepts/Curiosity Training.md"): f"""---
title: Curiosity Training
category: concepts
created: {DEMO_TIMESTAMP}
updated: {DEMO_TIMESTAMP}
sources:
  - raw/notes/question-trace-principles.md
---

# Curiosity Training

Curiosity training learns from the questions a researcher asks, not just the documents they ingest.
Lore records question traces and uses them to rank which follow-up questions would be valuable next.

## Context

The first version can be heuristic: look at gaps, recent activity, and the user's prior phrasing.
Once traces accumulate, a local model can be trained to sound more like the user and target more
useful follow-ups.

## Key Claims

1. Question traces are training data for taste, not just audit logs.
2. Gap targeting, style match, novelty, and specificity are enough to produce a practical first
   reward function.
3. Personalized suggestions should still degrade gracefully to heuristics before a checkpoint exists.

## Connections

- [[LLM Wiki]] provides the durable state that traces are grounded against.
- [[Active Memory]] decides which prior state the agent should inspect before generating questions.

## Sources

- raw/notes/question-trace-principles.md — motivation and reward signals for question traces

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
""",
    Path("meta/Demo Tour.md"): f"""---
title: Demo Tour
category: meta
created: {DEMO_TIMESTAMP}
updated: {DEMO_TIMESTAMP}
sources:
  - raw/articles/llm-wiki.md
  - raw/articles/model-context-protocol.md
  - raw/articles/active-memory.md
  - raw/notes/question-trace-principles.md
---

# Demo Tour

This seeded demo is meant to show Lore's workflow on a fresh clone in under two minutes.

## Context

Start with status, then search, then suggestions. The wiki is intentionally small but connected so
you can see how index-based retrieval, question traces, and follow-up prompts work together.

## Key Claims

1. A fresh clone should have enough structure to demonstrate search, health, and suggestions without
   asking the user to prepare source files first.
2. A seeded demo is only useful if it can be reset deterministically.
3. The first follow-up suggestions should work even before the local model is trained.

## Connections

- [[LLM Wiki]] explains the durable knowledge layer.
- [[Model Context Protocol]] shows how the same workflow reaches different clients.
- [[Active Memory]] explains why the agent should hydrate the right prior state.
- [[Curiosity Training]] explains where the follow-up questions come from.

## Sources

- raw/articles/llm-wiki.md — product framing
- raw/articles/model-context-protocol.md — tool surface
- raw/articles/active-memory.md — memory hydration
- raw/notes/question-trace-principles.md — training signal design

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
""",
    Path("_index.md"): """# Wiki Index

## Concepts
- [[LLM Wiki]] — Durable markdown memory that compounds across conversations.
- [[Model Context Protocol]] — Stable tool interface for using the same wiki across clients.
- [[Active Memory]] — Hydrate the agent with the right prior state before answering.
- [[Curiosity Training]] — Rank follow-up questions from traces, gaps, and style signals.

## Meta
- [[Demo Tour]] — Fast path for seeing Lore work on a fresh clone.
""",
    Path("_log.md"): """# Wiki Log

## [2026-04-14] demo | Seeded starter corpus
- Created: [[LLM Wiki]], [[Model Context Protocol]], [[Active Memory]], [[Curiosity Training]], [[Demo Tour]]
- Seeded 3 question traces
""",
}

DEMO_QUESTIONS = [
    "How is an LLM wiki different from classic RAG for personal research?",
    "Where does MCP help when the wiki already has a CLI?",
    "Which curiosity signal best captures my taste instead of generic usefulness?",
]

MANAGED_DIRS = (RAW_DIR, WIKI_DIR, DATA_DIR, OUTPUTS_DIR)

ARTICLE_CONNECTIONS = [
    ("mcp", "Model Context Protocol"),
    ("model context protocol", "Model Context Protocol"),
    ("wiki", "LLM Wiki"),
    ("markdown", "LLM Wiki"),
    ("memory", "Active Memory"),
    ("context", "Active Memory"),
    ("trace", "Curiosity Training"),
    ("question", "Curiosity Training"),
    ("suggest", "Curiosity Training"),
]


@dataclass
class DemoIngestResult:
    """Result payload for the hosted Lore demo flow."""

    article_path: str
    article_title: str
    raw_path: str
    raw_title: str
    article_content: str
    suggestions: list[dict]
    suggestion_mode: str
    action: str


def seed_demo(reset: bool = False) -> dict:
    """Seed a deterministic starter corpus for a fresh clone."""
    if reset:
        for directory in MANAGED_DIRS:
            shutil.rmtree(directory, ignore_errors=True)
    elif _workspace_has_user_content():
        raise RuntimeError(
            "Lore demo seeding expects a fresh clone. Re-run with --reset if you want to replace "
            "raw/, wiki/, data/, and outputs/."
        )

    _create_workspace()
    _write_sources()
    _write_articles()

    rebuild_all_backlinks()
    rebuild_index()

    for question in DEMO_QUESTIONS:
        capture_question_trace(question, build_wiki_state_summary())

    suggestions, mode = generate_suggestions_with_mode(n=3, prefer_daemon=False)
    suggestion_file = DATA_DIR / ".latest_suggestions"
    suggestion_file.parent.mkdir(parents=True, exist_ok=True)
    suggestion_file.write_text(
        json.dumps({"mode": mode, "suggestions": suggestions}, indent=2),
        encoding="utf-8",
    )

    return {
        "articles": len(
            [path for path in DEMO_ARTICLES if path.suffix == ".md" and not path.name.startswith("_")]
        ),
        "sources": len(DEMO_SOURCES),
        "traces": len(DEMO_QUESTIONS),
        "suggestion_mode": mode,
        "suggestions_cached": len(suggestions),
    }


def _workspace_has_user_content() -> bool:
    article_files = [path for path in WIKI_DIR.rglob("*.md") if not path.name.startswith("_")]
    if article_files:
        return True

    raw_files = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    if raw_files:
        return True

    data_files = [path for path in DATA_DIR.rglob("*") if path.is_file()]
    return bool(data_files)


def _create_workspace() -> None:
    for directory in (
        RAW_DIR / "articles",
        RAW_DIR / "notes",
        WIKI_DIR / "concepts",
        WIKI_DIR / "meta",
        OUTPUTS_DIR,
        DATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _write_sources() -> None:
    for relative_path, content in DEMO_SOURCES.items():
        target = RAW_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_articles() -> None:
    for relative_path, content in DEMO_ARTICLES.items():
        target = WIKI_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_demo_workspace() -> dict:
    """Seed the demo workspace only when it is still empty."""
    if _workspace_has_user_content():
        from lore.evolve.trajectory import get_question_trace_stats

        suggestions, mode = generate_suggestions_with_mode(n=3, prefer_daemon=False)
        trace_stats = get_question_trace_stats()
        return {
            "articles": len(
                [path for path in WIKI_DIR.rglob("*.md") if path.is_file() and not path.name.startswith("_")]
            ),
            "sources": len([path for path in RAW_DIR.rglob("*") if path.is_file()]),
            "traces": trace_stats.get("total", 0),
            "suggestion_mode": mode,
            "suggestions_cached": len(suggestions),
        }
    return seed_demo(reset=False)


def stage_uploaded_file(source_path: str | Path) -> Path:
    """Copy an uploaded file into raw/ so the demo can reference it durably."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Upload not found: {source}")

    suffix = source.suffix.lower()
    subdir = "papers" if suffix == ".pdf" else "notes"
    safe_stem = re.sub(r"[^\w.-]+", "-", source.stem).strip("-") or "uploaded-source"
    dest = RAW_DIR / subdir / f"{safe_stem}{suffix or '.txt'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def ingest_demo_source(source: str | Path, *, kind: str) -> DemoIngestResult:
    """
    Ingest one source into the demo workspace, create a single wiki page,
    and return follow-up suggestions.
    """
    ensure_demo_workspace()

    if kind == "url":
        ingested = ingest_url_result(str(source))
    elif kind == "file":
        staged = stage_uploaded_file(source)
        ingested = ingest_file_result(staged, force=True)
    else:
        raise ValueError(f"Unsupported demo source kind: {kind}")

    article_title = f"{_sanitize_title(ingested.title)} Notes"
    article_content = _build_summary_article(article_title, ingested)
    article_relpath = f"papers/{article_title}.md"
    article_path = WIKI_DIR / article_relpath

    if article_path.exists():
        action = "updated"
        _update_article(article_relpath, article_content)
    else:
        action = "created"
        _write_article("papers", article_title, article_content)

    rebuild_all_backlinks()
    rebuild_index()

    suggestions, mode = generate_suggestions_with_mode(n=3, prefer_daemon=False)
    return DemoIngestResult(
        article_path=f"wiki/{article_relpath}",
        article_title=article_title,
        raw_path=ingested.raw_path,
        raw_title=ingested.title,
        article_content=article_content,
        suggestions=suggestions,
        suggestion_mode=mode,
        action=action,
    )


def workspace_snapshot() -> dict:
    """Small status payload for the hosted demo UI."""
    article_files = sorted(
        path for path in WIKI_DIR.rglob("*.md") if path.is_file() and not path.name.startswith("_")
    )
    source_files = sorted(path for path in RAW_DIR.rglob("*") if path.is_file())
    latest_article = article_files[-1] if article_files else None

    return {
        "article_count": len(article_files),
        "source_count": len(source_files),
        "latest_article": str(latest_article.relative_to(WIKI_DIR)) if latest_article else None,
    }


def _sanitize_title(title: str) -> str:
    clean = re.sub(r"[\\/:*?\"<>|]+", " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or "Untitled Source"


def _build_summary_article(article_title: str, ingested: IngestedSource) -> str:
    now = datetime.now(timezone.utc).isoformat()
    summary = _lead_summary(ingested.content)
    claims = _key_claims(ingested.content)
    connections = _detect_connections(ingested.content)
    source_ref = _source_reference(ingested.raw_path)

    connection_lines = "\n".join(f"- [[{title}]]" for title in connections) or "- [[Demo Tour]]"
    claim_lines = "\n".join(f"{idx}. {claim}" for idx, claim in enumerate(claims, 1))

    return f"""---
title: {article_title}
category: papers
created: {now}
updated: {now}
sources:
  - {source_ref}
---

# {article_title}

{summary}

## Context

This page was created by the hosted Lore demo after ingesting one source. It is intentionally
deterministic: the demo writes a compact summary page, rebuilds the wiki index, and returns
follow-up questions without requiring a local checkpoint.

## Key Claims

{claim_lines}

## Connections

{connection_lines}

## Sources

- {source_ref} — ingested through the Lore demo workflow

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->
"""


def _lead_summary(content: str) -> str:
    sentences = _sentences(content, limit=2)
    if not sentences:
        return "This source was ingested into Lore, but the extracted text was too thin to summarize cleanly."
    return " ".join(sentences)


def _key_claims(content: str) -> list[str]:
    candidates = _sentences(content, limit=6)
    claims = []
    seen = set()
    for sentence in candidates:
        normalized = re.sub(r"\W+", "", sentence.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        claims.append(sentence)
        if len(claims) == 3:
            break
    if claims:
        return claims
    return [
        "The source adds one more durable artifact to the wiki instead of leaving the insight in transient chat history.",
        "A good demo page should make the source legible enough to connect it to the rest of the graph.",
        "Follow-up questions matter because they turn a single ingest into an ongoing research thread.",
    ]


def _sentences(content: str, limit: int) -> list[str]:
    text = re.sub(r"\s+", " ", content).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for part in parts:
        sentence = part.strip(" -")
        if len(sentence.split()) < 6:
            continue
        cleaned.append(sentence)
        if len(cleaned) >= limit:
            break
    return cleaned


def _detect_connections(content: str) -> list[str]:
    haystack = content.lower()
    connections = []
    for needle, title in ARTICLE_CONNECTIONS:
        if needle in haystack and title not in connections:
            connections.append(title)
    if not connections:
        connections.append("Demo Tour")
    return connections[:4]


def _source_reference(raw_path: str) -> str:
    path = Path(raw_path)
    try:
        return str(path.relative_to(WIKI_DIR.parent))
    except ValueError:
        return str(path)


def _extract_summary(content: str) -> str:
    in_frontmatter = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter or not stripped or stripped.startswith("#"):
            continue
        return stripped[:120]
    return "No summary available"


def _write_article(category: str, title: str, content: str) -> None:
    target_dir = WIKI_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{title}.md"

    target_path.write_text(snap_wikilinks(content).rstrip() + "\n", encoding="utf-8")
    _auto_update_index(title, _extract_summary(content), category)
    _auto_append_log(f"Created: [[{title}]]", action="write")


def _update_article(path: str, content: str) -> None:
    article_path = WIKI_DIR / path
    article_path.write_text(snap_wikilinks(content).rstrip() + "\n", encoding="utf-8")
    _auto_append_log(f"Updated: [[{article_path.stem}]]", action="update")


def _auto_update_index(title: str, summary: str, category: str) -> None:
    index_path = WIKI_DIR / "_index.md"
    entry = f"- [[{title}]] — {summary}"

    if not index_path.exists():
        index_path.write_text(f"# Wiki Index\n\n## {category.title()}\n{entry}\n", encoding="utf-8")
        return

    current = index_path.read_text(encoding="utf-8", errors="replace")
    if f"[[{title}]]" in current:
        return

    heading = f"## {category.title()}"
    if heading in current:
        current = current.replace(heading, f"{heading}\n{entry}", 1)
    else:
        current = current.rstrip() + f"\n\n{heading}\n{entry}\n"

    index_path.write_text(current, encoding="utf-8")


def _auto_append_log(detail: str, *, action: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"## [{now}] {action} | {detail}"

    log_path = WIKI_DIR / "_log.md"
    if not log_path.exists():
        log_path.write_text(f"# Wiki Log\n\n{entry}\n", encoding="utf-8")
        return

    current = log_path.read_text(encoding="utf-8", errors="replace")
    current = current.rstrip() + "\n\n" + entry + "\n"
    log_path.write_text(current, encoding="utf-8")
