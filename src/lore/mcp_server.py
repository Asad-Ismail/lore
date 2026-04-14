"""
Lore MCP Server — expose wiki capabilities to any MCP-compatible client.

Usage:
    HTTP (remote clients):        uv run lore-mcp --transport http --port 8766
    Stdio (Claude Desktop):       uv run lore-mcp --transport stdio
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

from lore.config import WIKI_DIR

INSTRUCTIONS = """\
You are maintaining a personal knowledge wiki called Lore. You have tools to \
search, read, write, and maintain interlinked wiki articles. The wiki lives \
in the wiki/ directory (viewable in Obsidian). Source documents live in raw/ \
(read-only — never modify sources).

## The Three Layers
1. raw/ — Immutable source documents. You read from here via ingest, never write.
2. wiki/ — Your wiki (Obsidian vault). You create, update, and maintain articles here.
3. These instructions — the schema that governs how the wiki works.

## Workflow: Answering Questions
1. Call read_article("_index.md") to see the full article catalog
2. Call search_and_read to find and read relevant articles
3. Synthesize an answer using [[WikiLink]] citations to the articles you used
4. File your answer back into the wiki automatically:
   - Comparison or analysis → write_article to create a new article
   - New fact about existing topic → update_article to add to Key Claims
   - Research gap or open question → update_article on "meta/Open Questions.md"
   - Trivial/conversational answers → don't write anything
   When in doubt, write it. It's cheap to delete, expensive to lose.
5. Call capture_trace with the user's question (trains the suggestion model)
6. Call generate_suggestions and show 2-3 follow-up questions at the end

## Workflow: Ingesting a URL
1. Call ingest_url — downloads content, saves to raw/, extracts text
2. Share key takeaways with the user, then WAIT for their response. \
   The user may want to guide what to emphasize or skip. Do NOT proceed \
   to writing articles until the user confirms.
3. Call write_article to create a summary page in papers/ (or appropriate category)
4. For each key concept/technique in the source:
   - If article exists → read it, call update_article with new information
   - If no article exists → call write_article to create one
   A single source typically touches 5-15 wiki articles.
5. Ensure all new articles link to related existing articles via [[WikiLinks]]. \
   Create [[WikiLink]] stubs for important referenced concepts even if no article \
   exists yet — they show as knowledge gaps in Obsidian's graph.
6. Call update_index to add entries for all new articles
7. Call append_log with: "## [YYYY-MM-DD] ingest | Source Title\\n- Created: [[...]]\\n- Updated: [[...]]"
8. Suggest which referenced sources are worth ingesting next.

## Workflow: Health Check / Lint
1. Call run_health_check for an automated scan
2. Review the findings and fix what you find:
   - Broken links → call cleanup_links to auto-fix
   - Orphan articles → call update_article on related articles to add inbound links
   - Stubs → call write_article to fill them by synthesizing from existing wiki content
   - Undiscovered connections → call update_article to add cross-references
   - Contradictions → update both articles with a ## Contradictions section
3. Call rebuild_index after fixes
4. Call append_log: "## [YYYY-MM-DD] lint | Fixed N issues"

## Article Format
Every wiki article must follow this structure:

---
title: Article Title
category: techniques
created: YYYY-MM-DDTHH:MM:SS+00:00
updated: YYYY-MM-DDTHH:MM:SS+00:00
sources:
  - raw/papers/source-name.md
---

# Article Title

2-sentence definition/summary.

## Context

Where this fits in the broader landscape.

## Key Claims

Numbered, citable claims from sources. Each claim should reference \
which source it comes from.

## Connections

How this relates to other wiki articles. Use [[WikiLink]] syntax.

## Sources

- [[source-paper]] — Section 3, relevant detail

## Referenced By

<!-- auto-maintained by cleanup_links — do not edit manually -->

## File Naming
Use the article title as the filename, preserving spaces and casing. \
Example: "wiki/concepts/Post-Training Quantization.md" (correct), \
NOT "wiki/concepts/post-training-quantization.md" (Obsidian can't resolve the wikilink).

## WikiLink Rules
- Use [[Article Title]] to link to other articles
- One canonical name per concept — don't create both [[AWQ]] and [[AWQ Paper]]
- If referencing a concept without an article, still use [[WikiLink]] — it becomes a stub
- When writing articles, the write_article tool auto-fixes broken links to closest matches

## Categories
concepts/ — core ideas, theory, overviews
techniques/ — specific methods and algorithms
papers/ — one summary page per source document
models/ — architecture or system summaries
datasets/ — dataset provenance and stats
benchmarks/ — evaluation benchmarks
people/ — profiles of key people
meta/ — reading lists, open questions, synthesis

Create new categories when existing ones get too broad.

## _index.md Format
Content-oriented catalog. Every article with a link and one-line summary:
- [[Quantization]] — Reducing numerical precision of model weights/activations (3 sources)
- [[GPTQ]] — Second-order weight quantization using Hessian information (2023-03)
Include source count or date to show article maturity.

## _log.md Format
Chronological, append-only. Each entry:
## [YYYY-MM-DD] ingest | Source Title
- Created: [[Article A]], [[Article B]]
- Updated: [[Article C]]

## Math and Diagrams
- Use LaTeX: $inline$ and $$block$$ (Obsidian renders via MathJax)
- Use Mermaid diagrams in ```mermaid code blocks (Obsidian renders natively)

## Important Rules
- raw/ is read-only. Never modify source documents.
- Link generously. The value of the wiki is in its connections.
- When new information contradicts existing articles, update BOTH and note the contradiction.
- Every ingest must update _index.md and _log.md.
- Always show follow-up suggestions at the end of your response.
- After writing or updating articles, call rebuild_index to keep search accurate.
"""

mcp = FastMCP("Lore Knowledge Wiki", instructions=INSTRUCTIONS)

MAX_INGEST_CHARS = 10_000


# ── Read tools ───────────────────────────────────────────────────────────────


@mcp.tool()
def read_article(path: str) -> str:
    """Read a wiki article by its relative path.

    Use "_index.md" to read the full table of contents.
    Use "_log.md" to read the operations log.
    Example paths: "concepts/Weight Quantization.md", "papers/AWQ.md"
    """
    article_path = (WIKI_DIR / path).resolve()
    if not article_path.is_relative_to(WIKI_DIR.resolve()):
        return "Access denied: path outside wiki directory."
    if not article_path.exists():
        return f"Article not found: {path}"
    return article_path.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def search_wiki(query: str, top_k: int = 5) -> str:
    """Search wiki articles using TF-IDF.

    Returns ranked results with titles, categories, relevance scores,
    and text snippets. Use this to find relevant articles before reading them.
    """
    from lore.index.search import search_wiki as _search, format_search_results

    results = _search(query, top_k=top_k)
    return format_search_results(results, show_snippets=True)


@mcp.tool()
def search_and_read(query: str, top_k: int = 3) -> str:
    """Search wiki and return full article content for top results.

    More expensive than search_wiki but gives you the complete text.
    Use when you need to synthesize an answer from multiple articles.
    """
    from lore.index.search import search_and_read as _search_and_read

    results = _search_and_read(query, top_k=top_k)
    if not results:
        return "No results found."

    parts = []
    for sr, content in results:
        truncated = content[:5000]
        if len(content) > 5000:
            truncated += f"\n[truncated — full article: {len(content):,} chars]"
        parts.append(
            f"## [{sr.category}] {sr.title} (score: {sr.score:.4f})\n"
            f"Path: wiki/{sr.article_path}\n\n"
            f"{truncated}"
        )
    return "\n\n---\n\n".join(parts)


# ── Helpers (auto-update index and log) ───────────────────────────────────────


def _extract_summary(content: str) -> str:
    """Pull a one-line summary from article content (first non-heading, non-frontmatter line)."""
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


def _auto_update_index(title: str, summary: str, category: str) -> None:
    """Append a new entry to _index.md under the right category heading."""
    index_path = WIKI_DIR / "_index.md"
    entry = f"- [[{title}]] — {summary}"

    if not index_path.exists():
        index_path.write_text(
            f"# Wiki Index\n\n## {category.title()}\n{entry}\n",
            encoding="utf-8",
        )
        return

    current = index_path.read_text(encoding="utf-8", errors="replace")

    # Check if already indexed
    if f"[[{title}]]" in current:
        return

    # Try to find the category heading and insert under it
    heading = f"## {category.title()}"
    if heading in current:
        # Insert after the heading line
        current = current.replace(heading, f"{heading}\n{entry}", 1)
    else:
        # Append new category section at the end
        current = current.rstrip() + f"\n\n{heading}\n{entry}\n"

    index_path.write_text(current, encoding="utf-8")


def _auto_append_log(detail: str, action: str = "write") -> None:
    """Append a timestamped entry to _log.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"## [{now}] {action} | {detail}"

    log_path = WIKI_DIR / "_log.md"
    if not log_path.exists():
        log_path.write_text(f"# Wiki Log\n\n{entry}\n", encoding="utf-8")
        return

    current = log_path.read_text(encoding="utf-8", errors="replace")
    current = current.rstrip() + "\n\n" + entry + "\n"
    log_path.write_text(current, encoding="utf-8")


# ── Write tools ──────────────────────────────────────────────────────────────


@mcp.tool()
def write_article(category: str, title: str, content: str) -> str:
    """Create a new wiki article.

    Automatically updates _index.md and _log.md so the article is
    immediately visible in Obsidian's catalog and activity log.

    Args:
        category: Subdirectory (concepts, techniques, papers, models, datasets,
                  benchmarks, people, meta).
        title: Article title — used as the filename (e.g. "Weight Quantization").
        content: Full markdown content including frontmatter.
               Use [[WikiLink]] syntax to link to other articles.
    """
    from lore.linker import snap_wikilinks

    cat_dir = WIKI_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    article_path = cat_dir / f"{title}.md"

    if article_path.exists():
        return f"Article already exists: {category}/{title}.md — use update_article instead."

    content = snap_wikilinks(content)
    article_path.write_text(content, encoding="utf-8")

    # Extract first sentence for index summary
    summary = _extract_summary(content)
    _auto_update_index(title, summary, category)
    _auto_append_log(f"Created: [[{title}]]", action="write")

    return f"Created: wiki/{category}/{title}.md ({len(content)} chars) — index and log updated"


@mcp.tool()
def update_article(path: str, content: str) -> str:
    """Replace the full content of an existing wiki article.

    Automatically logs the update to _log.md.

    Args:
        path: Relative path (e.g. "concepts/Weight Quantization.md").
        content: New full markdown content. Use [[WikiLink]] syntax.
    """
    from lore.linker import snap_wikilinks

    article_path = (WIKI_DIR / path).resolve()
    if not article_path.is_relative_to(WIKI_DIR.resolve()):
        return "Access denied: path outside wiki directory."
    if not article_path.exists():
        return f"Article not found: {path} — use write_article to create it."

    content = snap_wikilinks(content)
    article_path.write_text(content, encoding="utf-8")

    title = article_path.stem
    _auto_append_log(f"Updated: [[{title}]]", action="update")

    return f"Updated: wiki/{path} ({len(content)} chars) — log updated"


@mcp.tool()
def update_index(entries: str) -> str:
    """Append new entries to the wiki index (_index.md).

    Args:
        entries: Markdown lines to append, e.g.:
                "- [[New Article]] — One-line summary of this article"
                Add entries under the correct category heading.
                If you need to restructure, read the index first and
                provide the full replacement content.
    """
    index_path = WIKI_DIR / "_index.md"
    if not index_path.exists():
        index_path.write_text(f"# Wiki Index\n\n{entries}\n", encoding="utf-8")
        return "Created _index.md with new entries."

    current = index_path.read_text(encoding="utf-8", errors="replace")
    current = current.rstrip() + "\n" + entries + "\n"
    index_path.write_text(current, encoding="utf-8")
    return f"Appended to _index.md ({len(entries)} chars added)"


@mcp.tool()
def append_log(entry: str) -> str:
    """Append an entry to the wiki operations log (_log.md).

    Args:
        entry: Log entry, e.g.:
               "## [2026-04-14] ingest | Paper Title\\n- Created: [[Article]]\\n- Updated: [[Other]]"
    """
    log_path = WIKI_DIR / "_log.md"
    if not log_path.exists():
        log_path.write_text(f"# Wiki Log\n\n{entry}\n", encoding="utf-8")
        return "Created _log.md with first entry."

    current = log_path.read_text(encoding="utf-8", errors="replace")
    current = current.rstrip() + "\n\n" + entry + "\n"
    log_path.write_text(current, encoding="utf-8")
    return "Appended to _log.md"


@mcp.tool()
def cleanup_links() -> str:
    """Fix broken wikilinks and rebuild backlink footers across all articles.

    Snaps broken links to the closest matching article title and updates
    the 'Referenced By' section in every article. Logs the cleanup.
    """
    from lore.linker import rebuild_all_backlinks, snap_wikilinks

    fixed = 0
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        original = md_file.read_text(encoding="utf-8", errors="replace")
        snapped = snap_wikilinks(original)
        if snapped != original:
            md_file.write_text(snapped, encoding="utf-8")
            fixed += 1

    updated = rebuild_all_backlinks()

    _auto_append_log(
        f"Cleanup — {fixed} links fixed, {updated} backlinks rebuilt",
        action="cleanup",
    )

    return json.dumps({"links_fixed": fixed, "backlinks_updated": updated})


# ── Ingestion ────────────────────────────────────────────────────────────────


@mcp.tool()
def ingest_url(url: str) -> str:
    """Ingest a URL (paper, article, web page) into the Lore wiki.

    Downloads the content, saves it to raw/, extracts text, and registers
    a fingerprint for deduplication. For arXiv links, downloads the actual PDF.
    Automatically logs the ingestion to _log.md.
    Returns the extracted text (truncated).

    After ingesting, you should:
    1. Discuss key takeaways with the user
    2. Create wiki articles with write_article (auto-updates index and log)
    3. Update related existing articles with update_article (auto-updates log)
    """
    from lore.ingest.pipeline import ingest_url as _ingest_url

    text = _ingest_url(url)
    _auto_append_log(f"Ingested: {url}", action="ingest")

    if len(text) > MAX_INGEST_CHARS:
        return text[:MAX_INGEST_CHARS] + f"\n\n[truncated — full text: {len(text):,} chars]"
    return text


# ── Health & maintenance ─────────────────────────────────────────────────────


@mcp.tool()
def run_health_check() -> str:
    """Audit the wiki for broken links, orphan articles, stubs, and undiscovered connections.

    Also writes a health report to wiki/_meta/health_report.md and logs the check.
    Returns a JSON summary of all findings.
    """
    from lore.health.checker import run_health_check as _health_check

    results = _health_check()

    summary = {
        "broken_links": {k: v for k, v in list(results.get("broken_links", {}).items())[:20]},
        "orphans": results.get("orphans", [])[:20],
        "stubs": results.get("stubs", [])[:30],
        "connections": results.get("connections", [])[:15],
        "stats": results.get("stats", {}),
    }

    total_issues = (
        sum(len(v) for v in results.get("broken_links", {}).values())
        + len(results.get("orphans", []))
        + len(results.get("stubs", []))
    )
    _auto_append_log(
        f"Health check — {results.get('stats', {}).get('total_articles', 0)} articles, "
        f"{total_issues} issues found",
        action="health",
    )

    return json.dumps(summary, indent=2, default=str)


@mcp.tool()
def rebuild_index() -> str:
    """Rebuild the TF-IDF search index from all wiki articles.

    Run this after writing or updating articles to keep search accurate.
    """
    from lore.index.store import rebuild_index as _rebuild

    result = _rebuild()
    return json.dumps(result)


# ── Curiosity & training ─────────────────────────────────────────────────────


@mcp.tool()
def generate_suggestions(n: int = 3) -> str:
    """Generate curiosity-driven follow-up questions based on the wiki's state
    and the user's questioning patterns.

    Tries the daemon first (~100ms). If daemon is not running, falls back to
    loading the model directly (~17s). Show these at the end of every answer.
    """
    from lore.evolve.curiosity import generate_suggestions as _suggest

    suggestions = _suggest(n=n)
    if not suggestions:
        return "No suggestions available. The wiki may be empty or no model checkpoint exists."
    return json.dumps(suggestions, indent=2)


@mcp.tool()
def capture_trace(question: str) -> str:
    """Record a question trace for curiosity training.

    Call this after answering every substantive question. Captures the question
    along with the current wiki state. These traces train the local model to
    suggest better follow-up questions over time.
    """
    from lore.evolve.curiosity import build_wiki_state_summary
    from lore.evolve.trajectory import capture_question_trace, get_question_trace_stats

    wiki_state = build_wiki_state_summary()
    trace = capture_question_trace(question, wiki_state)
    stats = get_question_trace_stats()
    return json.dumps({
        "trace_id": trace.id[:8],
        "total_traces": stats["total"],
        "untrained_traces": stats["untrained"],
    })


# ── Status ───────────────────────────────────────────────────────────────────


@mcp.tool()
def get_status() -> str:
    """Get the overall status of the Lore wiki system.

    Returns article counts, ingestion stats, question trace stats,
    and whether the curiosity daemon is running.
    """
    from lore.ingest.pipeline import get_ingestion_stats
    from lore.index.store import get_index_stats
    from lore.evolve.trajectory import get_question_trace_stats
    from lore.evolve.curiosity import is_daemon_running

    return json.dumps({
        "ingestion": get_ingestion_stats(),
        "index": get_index_stats(),
        "traces": get_question_trace_stats(),
        "daemon_running": is_daemon_running(),
    }, indent=2)


# ── Resources ────────────────────────────────────────────────────────────────


@mcp.resource("lore://wiki/index")
def wiki_index_resource() -> str:
    """The wiki's table of contents — all articles organized by category."""
    index_path = WIKI_DIR / "_index.md"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8", errors="replace")
    return "No wiki index found. Run rebuild_index or ingest some sources first."


@mcp.resource("lore://wiki/article/{path}")
def wiki_article_resource(path: str) -> str:
    """Read a specific wiki article by its relative path."""
    article_path = (WIKI_DIR / path).resolve()
    if not article_path.is_relative_to(WIKI_DIR.resolve()):
        return "Access denied: path outside wiki directory."
    if not article_path.exists():
        return f"Article not found: {path}"
    return article_path.read_text(encoding="utf-8", errors="replace")


@mcp.resource("lore://wiki/health-report")
def wiki_health_report() -> str:
    """The most recent wiki health report."""
    report_path = WIKI_DIR / "_meta" / "health_report.md"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8", errors="replace")
    return "No health report found. Run the run_health_check tool first."


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    transport = "http"
    port = 8766
    host = "127.0.0.1"

    args = sys.argv[1:]
    if "--transport" in args:
        transport = args[args.index("--transport") + 1]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    if "--host" in args:
        host = args[args.index("--host") + 1]

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
