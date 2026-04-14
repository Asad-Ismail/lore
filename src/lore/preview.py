"""Preview helpers for browsing Lore pages and rendering a wiki graph."""

from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from lore.config import WIKI_DIR
from lore.index.store import load_all_articles
from lore.linker import extract_wikilinks

CATEGORY_ORDER = [
    "concepts",
    "papers",
    "models",
    "techniques",
    "datasets",
    "benchmarks",
    "people",
    "meta",
]

CATEGORY_COLORS = {
    "concepts": "#0d6b6f",
    "papers": "#c86a2d",
    "models": "#4d7c6c",
    "techniques": "#8a5a44",
    "datasets": "#6171a3",
    "benchmarks": "#8a4d72",
    "people": "#73663b",
    "meta": "#587191",
}


@dataclass(frozen=True)
class PreviewArticle:
    path: str
    title: str
    category: str
    content: str
    snippet: str
    incoming: tuple[str, ...]
    outgoing: tuple[str, ...]


def build_preview_articles(wiki_dir: Path = WIKI_DIR) -> list[PreviewArticle]:
    """Load wiki pages and derive a forward-link graph for preview UIs."""
    articles = load_all_articles(wiki_dir)
    article_lookup = {_normalize(article.title): article for article in articles}
    incoming_map = {article.path: set() for article in articles}
    outgoing_map: dict[str, list[str]] = {article.path: [] for article in articles}

    for article in articles:
        article_path = wiki_dir / article.path
        try:
            text = article_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = article.content
        text = _strip_frontmatter(text)
        text = _strip_backlink_footer(text)

        seen = set()
        for target in extract_wikilinks(text):
            matched = article_lookup.get(_normalize(target))
            if matched is None or matched.path == article.path or matched.path in seen:
                continue
            seen.add(matched.path)
            outgoing_map[article.path].append(matched.path)
            incoming_map[matched.path].add(article.path)

    return [
        PreviewArticle(
            path=article.path,
            title=article.title,
            category=article.category,
            content=article.content,
            snippet=article.snippet,
            incoming=tuple(sorted(incoming_map[article.path])),
            outgoing=tuple(outgoing_map[article.path]),
        )
        for article in articles
    ]


def preview_article_choices(preview_articles: list[PreviewArticle]) -> list[tuple[str, str]]:
    """Build Gradio dropdown choices for browsing preview articles."""
    return [
        (f"{article.title} · {article.category}", article.path)
        for article in preview_articles
    ]


def resolve_preview_path(
    preview_articles: list[PreviewArticle],
    preferred: str | None = None,
) -> str | None:
    """Return a valid article path for preview selection."""
    if not preview_articles:
        return None

    known_paths = {article.path for article in preview_articles}
    if preferred in known_paths:
        return preferred

    return preview_articles[-1].path


def render_library_markdown(
    preview_articles: list[PreviewArticle],
    selected_path: str | None = None,
) -> str:
    """Render a compact snapshot of the current wiki pages."""
    if not preview_articles:
        return "### Vault Snapshot\n\n_No pages yet._"

    selected_path = resolve_preview_path(preview_articles, selected_path)
    lines = [
        "### Vault Snapshot",
        "",
        "_Deterministic preview: browse the actual markdown pages and their wikilinks._",
        "",
    ]
    for article in preview_articles:
        marker = "->" if article.path == selected_path else "-"
        lines.append(
            f"{marker} `{article.path}` · {len(article.outgoing)} out · {len(article.incoming)} in"
        )
    return "\n".join(lines)


def render_article_markdown(
    preview_articles: list[PreviewArticle],
    selected_path: str | None = None,
) -> str:
    """Render the selected preview article with graph metadata."""
    selected = _select_article(preview_articles, selected_path)
    if selected is None:
        return "### Inspect Page\n\n_Select a page to inspect the markdown Lore generated._"

    article_map = {article.path: article for article in preview_articles}
    outgoing = _article_titles(selected.outgoing, article_map)
    incoming = _article_titles(selected.incoming, article_map)

    return (
        "### Inspect Page\n\n"
        f"- Path: `{selected.path}`\n"
        f"- Category: **{selected.category}**\n"
        f"- Links out: **{len(selected.outgoing)}**\n"
        f"- Referenced by: **{len(selected.incoming)}**\n"
        f"- Forward links: {outgoing}\n"
        f"- Backlinks: {incoming}\n\n"
        "---\n\n"
        f"{selected.content}"
    )


def render_graph_html(
    preview_articles: list[PreviewArticle],
    selected_path: str | None = None,
) -> str:
    """Render the preview wiki as a deterministic SVG graph."""
    selected = _select_article(preview_articles, selected_path)
    if selected is None:
        return (
            "<div class='graph-shell'>"
            "<div class='graph-summary'><strong>No graph yet.</strong>"
            "<span>Ingest a source to create a page.</span></div></div>"
        )

    article_map = {article.path: article for article in preview_articles}
    categories = _present_categories(preview_articles)
    nodes_by_category = {
        category: [article for article in preview_articles if article.category == category]
        for category in categories
    }

    max_nodes = max(len(nodes) for nodes in nodes_by_category.values())
    width = max(920, 180 + len(categories) * 220)
    height = max(420, 210 + max_nodes * 96)
    positions: dict[str, tuple[float, float]] = {}

    column_step = (width - 160) / max(1, len(categories))
    inner_height = height - 190
    for idx, category in enumerate(categories):
        nodes = nodes_by_category[category]
        x = 100 + idx * column_step
        if len(nodes) == 1:
            positions[nodes[0].path] = (x, height / 2)
            continue

        gap = min(120, inner_height / max(1, len(nodes) - 1))
        used = gap * (len(nodes) - 1)
        start = 130 + (inner_height - used) / 2
        for node_idx, article in enumerate(nodes):
            positions[article.path] = (x, start + node_idx * gap)

    neighbor_paths = set(selected.incoming) | set(selected.outgoing)
    edge_count = sum(len(article.outgoing) for article in preview_articles)
    column_blocks = []
    for idx, category in enumerate(categories):
        color = CATEGORY_COLORS.get(category, "#8b8b8b")
        x = 100 + idx * column_step
        label = html.escape(category.replace("-", " ").title())
        column_blocks.append(
            f"<rect x='{x - 92:.1f}' y='58' width='184' height='{height - 118:.1f}' "
            f"rx='28' fill='{color}' opacity='0.08'/>"
        )
        column_blocks.append(
            f"<text x='{x:.1f}' y='92' text-anchor='middle' class='graph-column'>{label}</text>"
        )

    edge_blocks = []
    for article in preview_articles:
        x1, y1 = positions[article.path]
        for target_path in article.outgoing:
            x2, y2 = positions[target_path]
            if selected.path in {article.path, target_path}:
                stroke = "#0d6b6f"
                opacity = "0.82"
                width_px = "3.0"
            elif {article.path, target_path} & neighbor_paths:
                stroke = "#c86a2d"
                opacity = "0.35"
                width_px = "2.0"
            else:
                stroke = "#7a817a"
                opacity = "0.18"
                width_px = "1.4"
            dx = max(64, abs(x2 - x1) * 0.35)
            edge_blocks.append(
                "<path "
                f"d='M {x1:.1f} {y1:.1f} C {x1 + dx:.1f} {y1:.1f}, {x2 - dx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}' "
                f"stroke='{stroke}' stroke-width='{width_px}' stroke-opacity='{opacity}' "
                "fill='none' stroke-linecap='round'/>"
            )

    node_blocks = []
    for article in preview_articles:
        x, y = positions[article.path]
        color = CATEGORY_COLORS.get(article.category, "#8b8b8b")
        is_selected = article.path == selected.path
        is_neighbor = article.path in neighbor_paths
        if is_selected:
            fill = color
            text_color = "#fffaf2"
            stroke = color
        elif is_neighbor:
            fill = "#f3e2c8"
            text_color = "#13231a"
            stroke = color
        else:
            fill = "#fffaf1"
            text_color = "#13231a"
            stroke = color

        lines = _label_lines(article.title)
        node_blocks.append(
            f"<rect x='{x - 78:.1f}' y='{y - 28:.1f}' width='156' height='56' rx='18' "
            f"fill='{fill}' stroke='{stroke}' stroke-width='1.8'/>"
        )
        node_blocks.append(
            f"<text x='{x:.1f}' y='{y - 3:.1f}' text-anchor='middle' fill='{text_color}' class='graph-node'>"
        )
        for idx, line in enumerate(lines):
            dy = "-0.1em" if idx == 0 and len(lines) > 1 else ("1.05em" if idx == 1 else "0")
            node_blocks.append(
                f"<tspan x='{x:.1f}' dy='{dy}'>{html.escape(line)}</tspan>"
            )
        node_blocks.append("</text>")

    focus_title = html.escape(selected.title)
    focus_outgoing = html.escape(_article_titles(selected.outgoing, article_map, as_text=True))
    focus_incoming = html.escape(_article_titles(selected.incoming, article_map, as_text=True))

    return (
        "<div class='graph-shell'>"
        "<div class='graph-summary'>"
        "<strong>Deterministic graph preview</strong>"
        f"<span>{len(preview_articles)} pages</span>"
        f"<span>{edge_count} forward links</span>"
        f"<span>Focus: {focus_title}</span>"
        "</div>"
        "<div class='graph-detail'>"
        f"<span class='graph-pill'><strong>Outgoing</strong> {focus_outgoing}</span>"
        f"<span class='graph-pill'><strong>Backlinks</strong> {focus_incoming}</span>"
        "</div>"
        "<div class='graph-scroll'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Lore wiki graph'>"
        f"{''.join(column_blocks)}"
        f"{''.join(edge_blocks)}"
        f"{''.join(node_blocks)}"
        "</svg>"
        "</div>"
        "</div>"
    )


def _select_article(
    preview_articles: list[PreviewArticle],
    selected_path: str | None,
) -> PreviewArticle | None:
    resolved = resolve_preview_path(preview_articles, selected_path)
    if resolved is None:
        return None
    return next((article for article in preview_articles if article.path == resolved), None)


def _present_categories(preview_articles: list[PreviewArticle]) -> list[str]:
    categories = {article.category for article in preview_articles}
    ordered = [category for category in CATEGORY_ORDER if category in categories]
    ordered.extend(sorted(categories - set(ordered)))
    return ordered


def _article_titles(
    paths: tuple[str, ...],
    article_map: dict[str, PreviewArticle],
    *,
    as_text: bool = False,
) -> str:
    if not paths:
        return "_none_" if not as_text else "none"

    titles = [article_map[path].title for path in paths if path in article_map]
    if as_text:
        return ", ".join(titles)
    return ", ".join(f"`{title}`" for title in titles)


def _label_lines(title: str) -> list[str]:
    wrapped = textwrap.wrap(title, width=18, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return ["Untitled"]
    if len(wrapped) == 1:
        return wrapped
    if len(wrapped) > 2:
        second = wrapped[1]
        if len(second) > 16:
            second = second[:15].rstrip() + "..."
        else:
            second = second.rstrip() + "..."
        return [wrapped[0], second]
    return wrapped[:2]


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)


def _strip_backlink_footer(text: str) -> str:
    return re.sub(r"\n## Referenced By\n.*$", "", text, flags=re.DOTALL)


def _normalize(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value.lower()).strip()
