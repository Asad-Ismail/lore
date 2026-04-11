"""TF-IDF search index over wiki articles."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lore.config import WIKI_DIR, TFIDF_INDEX_PATH


@dataclass
class WikiArticle:
    path: str
    title: str
    category: str
    content: str
    snippet: str


def load_all_articles(wiki_dir: Path = WIKI_DIR) -> list[WikiArticle]:
    """Load all non-meta wiki articles."""
    articles = []
    for md_file in sorted(wiki_dir.rglob("*.md")):
        if md_file.name.startswith("_"):
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL).strip()
        title = _extract_title(text) or md_file.stem.replace("-", " ").title()
        category = md_file.parent.name
        snippet = content[:300].replace("\n", " ")

        articles.append(WikiArticle(
            path=str(md_file.relative_to(wiki_dir)),
            title=title,
            category=category,
            content=content,
            snippet=snippet,
        ))
    return articles


def _extract_title(text: str) -> Optional[str]:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def build_tfidf_index(articles: list[WikiArticle]) -> tuple:
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [f"{a.title} {a.title} {a.content}" for a in articles]
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(corpus)

    TFIDF_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_INDEX_PATH, "wb") as f:
        pickle.dump((vectorizer, matrix, [a.path for a in articles]), f)

    print(f"[index] TF-IDF: {matrix.shape[0]} docs, {matrix.shape[1]} features")
    return vectorizer, matrix, articles


def tfidf_search(query: str, top_k: int = 8) -> list[tuple[str, float]]:
    idx = load_tfidf_index()
    if idx is None:
        return []
    vectorizer, matrix, paths = idx
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(paths[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def load_tfidf_index():
    if not TFIDF_INDEX_PATH.exists():
        return None
    with open(TFIDF_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def rebuild_index() -> dict:
    articles = load_all_articles()
    if not articles:
        print("[index] No articles found.")
        return {"articles": 0}
    build_tfidf_index(articles)
    return {"articles": len(articles)}


def get_index_stats() -> dict:
    stats = {}
    if TFIDF_INDEX_PATH.exists():
        with open(TFIDF_INDEX_PATH, "rb") as f:
            _, matrix, paths = pickle.load(f)
        stats["tfidf_articles"] = len(paths)
    return stats
