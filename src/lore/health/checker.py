"""Wiki health checks: contradictions, stubs, orphans, connections."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lore.config import WIKI_DIR
from lore.linker import find_broken_links, find_orphan_articles, build_backlink_map
from lore.titles import path_to_title


def run_health_check(wiki_dir: Path = WIKI_DIR) -> dict:
    """
    Run all health checks and return results dict.
    Also writes wiki/_meta/health_report.md.
    """
    print("[health] Running health checks...")
    results = {}

    # 1. Broken links
    results["broken_links"] = find_broken_links(wiki_dir)
    print(f"[health] Broken links in {len(results['broken_links'])} articles")

    # 2. Orphan articles
    results["orphans"] = find_orphan_articles(wiki_dir)
    print(f"[health] Orphan articles: {len(results['orphans'])}")

    # 3. Stub candidates (wikilinks without matching article)
    results["stubs"] = _find_stub_candidates(wiki_dir)
    print(f"[health] Stub candidates: {len(results['stubs'])}")

    # 4. Similar article pairs (potential duplicates or undiscovered connections)
    results["connections"] = _find_undiscovered_connections(wiki_dir)
    print(f"[health] Undiscovered connections: {len(results['connections'])}")

    # 5. Article stats
    results["stats"] = _compute_stats(wiki_dir)

    # Write report
    report_path = wiki_dir / "_meta" / "health_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_report(results), encoding="utf-8")
    print(f"[health] Report written: {report_path}")

    return results


def _find_stub_candidates(wiki_dir: Path) -> list[str]:
    """Return [[WikiLink]] targets that have no corresponding article file."""
    existing_titles = {
        re.sub(r"[^\w\s]", "", path_to_title(f)).lower().strip()
        for f in wiki_dir.rglob("*.md")
        if not f.name.startswith("_")
    }

    stubs: set[str] = set()
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for link in re.findall(r"\[\[([^\]|#]+?)\]\]", content):
            normalized = re.sub(r"[^\w\s]", "", link.lower()).strip()
            if normalized not in existing_titles:
                stubs.add(link)

    return sorted(stubs)


def _find_undiscovered_connections(wiki_dir: Path, max_pairs: int = 20) -> list[dict]:
    """
    Find article pairs with semantic similarity but no explicit link.
    Uses TF-IDF cosine similarity as a proxy.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    articles = []
    contents = []
    for md_file in sorted(wiki_dir.rglob("*.md")):
        if md_file.name.startswith("_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        except Exception:
            continue
        articles.append(md_file)
        contents.append(content[:1000])

    if len(articles) < 2:
        return []

    # Build TF-IDF in-memory (separate from main index)
    vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vect.fit_transform(contents)

    # Compute pairwise similarity for nearby articles only (efficiency)
    import numpy as np
    connections = []
    checked = set()

    for i in range(len(articles)):
        sims = cosine_similarity(matrix[i], matrix).flatten()
        top_j = [j for j in sims.argsort()[::-1] if j != i and 0.25 < sims[j] < 0.75][:5]

        for j in top_j:
            pair_key = tuple(sorted([i, j]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            # Check if they already link to each other
            content_i = contents[i]
            content_j = contents[j]
            title_i = path_to_title(articles[i])
            title_j = path_to_title(articles[j])

            already_linked = (
                title_j.lower() in content_i.lower()
                or title_i.lower() in content_j.lower()
            )
            if not already_linked:
                connections.append({
                    "article_a": str(articles[i].relative_to(wiki_dir)),
                    "article_b": str(articles[j].relative_to(wiki_dir)),
                    "similarity": round(float(sims[j]), 3),
                    "title_a": title_i,
                    "title_b": title_j,
                })

    # Return top connections by similarity
    connections.sort(key=lambda x: x["similarity"], reverse=True)
    return connections[:max_pairs]


def _compute_stats(wiki_dir: Path) -> dict:
    """Count articles, words, backlinks."""
    total_articles = 0
    total_words = 0
    category_counts: dict[str, int] = defaultdict(int)

    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            content_clean = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        except Exception:
            continue
        total_articles += 1
        total_words += len(content_clean.split())
        category_counts[md_file.parent.name] += 1

    backlinks = build_backlink_map(wiki_dir)
    top_linked = sorted(backlinks.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    return {
        "total_articles": total_articles,
        "total_words": total_words,
        "by_category": dict(category_counts),
        "top_linked": [(title, len(refs)) for title, refs in top_linked],
    }


def _format_report(results: dict) -> str:
    now = datetime.now(timezone.utc)
    stats = results.get("stats", {})

    sections = [
        f"# Wiki Health Report",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"## Stats",
        f"- Total articles: {stats.get('total_articles', 0)}",
        f"- Total words: {stats.get('total_words', 0):,}",
        f"- By category:",
    ]
    for cat, count in sorted(stats.get("by_category", {}).items()):
        sections.append(f"  - {cat}: {count}")

    sections += ["", "## Top Linked Articles", ""]
    for title, count in stats.get("top_linked", []):
        sections.append(f"- **{title}** ← {count} articles")

    # Broken links
    broken = results.get("broken_links", {})
    sections += ["", f"## Broken Links ({sum(len(v) for v in broken.values())} total)", ""]
    if broken:
        for src, links in list(broken.items())[:20]:
            sections.append(f"- `{src}`: {', '.join(f'[[{l}]]' for l in links)}")
    else:
        sections.append("_None found_")

    # Orphans
    orphans = results.get("orphans", [])
    sections += ["", f"## Orphan Articles ({len(orphans)} total)", ""]
    if orphans:
        for path in orphans[:20]:
            sections.append(f"- `{path}`")
    else:
        sections.append("_None found_")

    # Stubs
    stubs = results.get("stubs", [])
    sections += ["", f"## Stub Candidates ({len(stubs)} total)", ""]
    if stubs:
        for stub in stubs[:30]:
            sections.append(f"- [[{stub}]]")
    else:
        sections.append("_None found_")

    # Undiscovered connections
    connections = results.get("connections", [])
    sections += ["", f"## Undiscovered Connections ({len(connections)} candidates)", ""]
    if connections:
        sections.append("These article pairs have high semantic similarity but no explicit link:")
        sections.append("")
        for conn in connections[:15]:
            sections.append(
                f"- [[{conn['title_a']}]] ↔ [[{conn['title_b']}]] "
                f"(similarity: {conn['similarity']})"
            )
    else:
        sections.append("_None found_")

    return "\n".join(sections) + "\n"
