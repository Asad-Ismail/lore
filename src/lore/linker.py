"""Build and maintain the [[WikiLink]] backlink graph."""

from __future__ import annotations

import re
from pathlib import Path

from lore.config import WIKI_DIR
from lore.titles import path_to_title


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[WikiLink]] targets from article content."""
    return re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]", content)


def build_backlink_map(wiki_dir: Path = WIKI_DIR) -> dict[str, list[str]]:
    """
    Build a map of article_name → [articles that link to it].
    Returns: {"LoRA": ["Quantization", "Fine-tuning Overview"], ...}
    """
    backlinks: dict[str, list[str]] = {}
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        source_title = _file_to_title(md_file)
        for target in extract_wikilinks(content):
            target_norm = _normalize_title(target)
            if target_norm not in backlinks:
                backlinks[target_norm] = []
            if source_title not in backlinks[target_norm]:
                backlinks[target_norm].append(source_title)
    return backlinks


def inject_backlinks(article_path: Path, backlinks: list[str]) -> None:
    """
    Inject or update the 'Referenced by' footer section in an article.
    """
    if not backlinks:
        return

    content = article_path.read_text(encoding="utf-8", errors="replace")
    footer_marker = "## Referenced By"
    new_footer = footer_marker + "\n" + "\n".join(
        f"- [[{title}]]" for title in sorted(backlinks)
    )

    if footer_marker in content:
        # Replace existing footer
        content = re.sub(
            r"## Referenced By\n.*$",
            new_footer,
            content,
            flags=re.DOTALL,
        )
    else:
        content = content.rstrip() + "\n\n" + new_footer + "\n"

    article_path.write_text(content, encoding="utf-8")


def find_broken_links(wiki_dir: Path = WIKI_DIR) -> dict[str, list[str]]:
    """
    Return map of source_file → [broken link targets].
    A link is broken if no wiki file matches the target title.
    """
    existing_titles = {
        _normalize_title(_file_to_title(f))
        for f in wiki_dir.rglob("*.md")
        if not f.name.startswith("_")
    }

    broken: dict[str, list[str]] = {}
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_") or any(p.name.startswith("_") for p in md_file.parents):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        bad = [
            t for t in extract_wikilinks(content)
            if _normalize_title(t) not in existing_titles
        ]
        if bad:
            broken[str(md_file.relative_to(wiki_dir))] = bad
    return broken


def find_orphan_articles(wiki_dir: Path = WIKI_DIR) -> list[str]:
    """Return list of article paths with no incoming backlinks."""
    backlinks = build_backlink_map(wiki_dir)
    orphans = []
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        title_norm = _normalize_title(_file_to_title(md_file))
        if title_norm not in backlinks or not backlinks[title_norm]:
            orphans.append(str(md_file.relative_to(wiki_dir)))
    return orphans


def rebuild_all_backlinks(wiki_dir: Path = WIKI_DIR) -> int:
    """Rebuild backlink footers in all wiki articles. Returns count updated."""
    backlinks = build_backlink_map(wiki_dir)
    updated = 0
    for md_file in wiki_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        title_norm = _normalize_title(_file_to_title(md_file))
        refs = backlinks.get(title_norm, [])
        if refs:
            inject_backlinks(md_file, refs)
            updated += 1
    return updated


def snap_wikilinks(content: str, wiki_dir: Path = WIKI_DIR) -> str:
    """
    Replace every [[Link]] in content with the closest matching existing
    article title. Called before writing any article to disk so broken
    links never reach the wiki in the first place.

    Strategy (in order):
      1. Exact normalized match         [[GPTQ]] → exists → keep
      2. Strip trailing punctuation     [[The Era of 1-bit LLMs:]] → [[The Era Of 1 Bit Llms]]
      3. Substring match (target ⊆ title or title ⊆ target)
      4. Word-overlap ratio ≥ 0.5
      5. No match → strip the link, keep display text bare
    """
    # Build lookup: normalized_title → display title (Title Case from filename)
    title_map: dict[str, str] = {}
    for f in wiki_dir.rglob("*.md"):
        if f.name.startswith("_") or any(p.name.startswith("_") for p in f.parents):
            continue
        display = _file_to_title(f)
        title_map[_normalize_title(display)] = display

    def _best_match(raw: str) -> str | None:
        norm = _normalize_title(raw)

        # 1. Exact
        if norm in title_map:
            return title_map[norm]

        # 2. Strip trailing/leading punctuation and retry
        stripped = _normalize_title(re.sub(r"^[\W_]+|[\W_]+$", "", raw))
        if stripped and stripped in title_map:
            return title_map[stripped]

        # 3. Substring match
        for key, display in title_map.items():
            if norm in key or key in norm:
                return display

        # 4. Word-overlap ≥ 0.5
        norm_words = set(norm.split())
        if norm_words:
            best_score, best_display = 0.0, None
            for key, display in title_map.items():
                key_words = set(key.split())
                if not key_words:
                    continue
                overlap = len(norm_words & key_words) / max(len(norm_words), len(key_words))
                if overlap > best_score:
                    best_score, best_display = overlap, display
            if best_score >= 0.5:
                return best_display

        return None  # No match — caller will strip the link

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        matched = _best_match(raw)
        if matched:
            return f"[[{matched}]]"
        # No match: keep as plain text (becomes a stub candidate, not a broken link)
        return raw

    return re.sub(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]", _replace, content)


def _file_to_title(path: Path) -> str:
    """Convert a wiki file path to a display title without rewriting acronym casing."""
    return path_to_title(path)


def _normalize_title(title: str) -> str:
    """Normalize a title for comparison (lowercase, no special chars)."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()
