"""Hybrid search: Reciprocal Rank Fusion of TF-IDF + embedding results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lore.config import WIKI_DIR, SEARCH_TOP_K, SEARCH_RRF_K, SEARCH_RRF_ALPHA
from lore.index.store import tfidf_search, embedding_search, load_all_articles, WikiArticle


@dataclass
class SearchResult:
    article_path: str
    title: str
    category: str
    snippet: str
    score: float
    tfidf_rank: int
    embed_rank: int


def hybrid_search(query: str, top_k: int = SEARCH_TOP_K) -> list[SearchResult]:
    """
    Hybrid search combining TF-IDF and embedding results via RRF.

    RRF score: alpha/(k + tfidf_rank) + (1-alpha)/(k + embed_rank)
    Higher = more relevant. Articles not appearing in one ranking get rank=infinity.
    """
    # Get ranked results from both methods
    tfidf_results = tfidf_search(query, top_k=top_k * 2)
    embed_results = embedding_search(query, top_k=top_k * 2)

    # Build rank maps
    tfidf_rank: dict[str, int] = {path: i + 1 for i, (path, _) in enumerate(tfidf_results)}
    embed_rank: dict[str, int] = {path: i + 1 for i, (path, _) in enumerate(embed_results)}

    # Union of all candidates
    all_paths = set(tfidf_rank) | set(embed_rank)

    # Compute RRF scores
    INF = top_k * 100  # Effective infinity for missing rank
    rrf_scores: dict[str, float] = {}
    for path in all_paths:
        tr = tfidf_rank.get(path, INF)
        er = embed_rank.get(path, INF)
        rrf_scores[path] = (
            SEARCH_RRF_ALPHA / (SEARCH_RRF_K + tr)
            + (1 - SEARCH_RRF_ALPHA) / (SEARCH_RRF_K + er)
        )

    # Sort and take top_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Load article metadata
    article_map = {a.path: a for a in load_all_articles()}

    results = []
    for path, score in ranked:
        article = article_map.get(path)
        if article is None:
            continue
        results.append(SearchResult(
            article_path=path,
            title=article.title,
            category=article.category,
            snippet=article.snippet,
            score=score,
            tfidf_rank=tfidf_rank.get(path, INF),
            embed_rank=embed_rank.get(path, INF),
        ))

    return results


def search_and_read(query: str, top_k: int = SEARCH_TOP_K) -> list[tuple[SearchResult, str]]:
    """
    Search and return full article content for each result.
    Returns [(result, full_content), ...]
    """
    results = hybrid_search(query, top_k=top_k)
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
    """Format search results for CLI display."""
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
