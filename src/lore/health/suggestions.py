"""
Wiki enhancement suggestions.

Separate from checker.py (which finds *problems*); this module finds
*opportunities* — article pairs worth connecting, concepts worth expanding,
and research questions the wiki could answer better.
"""

from __future__ import annotations

from pathlib import Path

from lore.config import WIKI_DIR
from lore.health.checker import _find_undiscovered_connections, _compute_stats


def suggest_connections(wiki_dir: Path = WIKI_DIR, max_suggestions: int = 20) -> list[dict]:
    """
    Article pairs with high semantic similarity but no explicit [[WikiLink]].
    These are the highest-value additions: one sentence linking two articles
    can dramatically improve Q&A recall.
    """
    return _find_undiscovered_connections(wiki_dir, max_pairs=max_suggestions)


def suggest_new_articles(wiki_dir: Path = WIKI_DIR) -> list[str]:
    """
    Concepts mentioned enough times across articles to deserve their own page.
    Returns concept names sorted by mention frequency.
    """
    import re
    from collections import Counter

    mention_counts: Counter = Counter()
    existing_titles = {
        re.sub(r"[^\w\s]", "", f.stem.replace("-", " ").title()).lower().strip()
        for f in wiki_dir.rglob("*.md")
        if not f.name.startswith("_")
    }

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_") or any(p.name.startswith("_") for p in md_file.parents):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for link in re.findall(r"\[\[([^\]|#]+?)\]\]", content):
            norm = re.sub(r"[^\w\s]", "", link.lower()).strip()
            if norm not in existing_titles:
                mention_counts[link] += 1

    # Return concepts mentioned ≥ 2 times, most frequent first
    return [concept for concept, count in mention_counts.most_common() if count >= 2]


def suggest_research_questions(wiki_dir: Path = WIKI_DIR) -> list[str]:
    """
    Generate research questions from gaps between article Connections sections.
    Looks for concepts mentioned in Connections but not yet in the wiki.
    """
    import re

    questions = []
    existing = {
        re.sub(r"[^\w\s]", "", f.stem.replace("-", " ").title()).lower().strip()
        for f in wiki_dir.rglob("*.md")
        if not f.name.startswith("_")
    }

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_") or any(p.name.startswith("_") for p in md_file.parents):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = md_file.stem.replace("-", " ").title()

        # Find concepts mentioned in Connections section but not in wiki
        connections_m = re.search(r"## Connections(.*?)(?=##|\Z)", content, re.DOTALL)
        if not connections_m:
            continue
        conn_text = connections_m.group(1)
        for link in re.findall(r"\[\[([^\]|#]+?)\]\]", conn_text):
            norm = re.sub(r"[^\w\s]", "", link.lower()).strip()
            if norm not in existing:
                questions.append(f"What is {link}, and how does it relate to {title}?")

    return list(dict.fromkeys(questions))  # deduplicate preserving order


def format_suggestions_report(wiki_dir: Path = WIKI_DIR) -> str:
    """Format all suggestions as a markdown report section."""
    connections = suggest_connections(wiki_dir)
    new_articles = suggest_new_articles(wiki_dir)
    questions = suggest_research_questions(wiki_dir)

    lines = ["## Enhancement Suggestions", ""]

    lines += [f"### Undiscovered Connections ({len(connections)})", ""]
    if connections:
        for c in connections[:10]:
            lines.append(f"- Add a link between [[{c['title_a']}]] and [[{c['title_b']}]] "
                         f"(similarity: {c['similarity']})")
    else:
        lines.append("_None_")

    lines += ["", f"### New Article Candidates ({len(new_articles)})", ""]
    if new_articles:
        for concept in new_articles[:15]:
            lines.append(f"- [[{concept}]]")
    else:
        lines.append("_None_")

    lines += ["", f"### Open Research Questions ({len(questions)})", ""]
    if questions:
        for q in questions[:10]:
            lines.append(f"- {q}")
    else:
        lines.append("_None_")

    return "\n".join(lines)
