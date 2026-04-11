"""TF-IDF search over wiki articles."""

from __future__ import annotations

from dataclasses import dataclass

from lore.config import WIKI_DIR, SEARCH_TOP_K
from lore.index.store import tfidf_search, load_all_articles, WikiArticle


@dataclass
class SearchResult:
    article_path: str
    title: str
    category: str
    snippet: str
    score: float


def search_wiki(query: str, top_k: int = SEARCH_TOP_K) -> list[SearchResult]:
    """Search wiki articles with TF-IDF."""
    results = tfidf_search(query, top_k=top_k)
    article_map = {a.path: a for a in load_all_articles()}

    output = []
    for path, score in results:
        article = article_map.get(path)
        if article is None:
            continue
        output.append(SearchResult(
            article_path=path,
            title=article.title,
            category=article.category,
            snippet=article.snippet,
            score=score,
        ))
    return output


def search_and_read(query: str, top_k: int = SEARCH_TOP_K) -> list[tuple[SearchResult, str]]:
    """Search and return full article content for each result."""
    results = search_wiki(query, top_k=top_k)
    output = []
    for r in results:
        full_path = WIKI_DIR / r.article_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = r.snippet
        output.append((r, content))
    return output


def format_search_results(results: list[SearchResult], show_snippets: bool = True) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r.category}] **{r.title}** (score: {r.score:.4f})")
        lines.append(f"   Path: wiki/{r.article_path}")
        if show_snippets:
            lines.append(f"   {r.snippet[:150]}...")
        lines.append("")
    return "\n".join(lines)
