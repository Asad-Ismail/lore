"""TF-IDF + embedding index over all wiki articles."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sqlitedict import SqliteDict

from lore.config import (
    WIKI_DIR, DATA_DIR, TFIDF_INDEX_PATH, EMBEDDINGS_DB,
    LORA_BASE_MODEL_ID, HF_CACHE_DIR,
)


@dataclass
class WikiArticle:
    path: str           # relative to WIKI_DIR
    title: str
    category: str
    content: str
    snippet: str        # first 300 chars of content (no frontmatter)


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

        # Strip frontmatter
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


# ── TF-IDF index ──────────────────────────────────────────────────────────────

def build_tfidf_index(articles: list[WikiArticle]) -> tuple:
    """Build and save TF-IDF index. Returns (vectorizer, matrix, article_list)."""
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


def load_tfidf_index():
    """Load saved TF-IDF index. Returns (vectorizer, matrix, paths) or None."""
    if not TFIDF_INDEX_PATH.exists():
        return None
    with open(TFIDF_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def tfidf_search(query: str, top_k: int = 8) -> list[tuple[str, float]]:
    """
    Search with TF-IDF. Returns [(article_path, score), ...] sorted by score.
    """
    idx = load_tfidf_index()
    if idx is None:
        return []
    vectorizer, matrix, paths = idx
    q_vec = vectorizer.transform([query])
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(paths[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# ── Embedding index ───────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Delegate to embedder module (single model-load point for the process)."""
    from lore.index.embedder import embed
    return embed(texts, batch_size=batch_size)


def build_embedding_index(articles: list[WikiArticle]) -> None:
    """Build and store embedding vectors for all wiki articles."""
    texts = [f"{a.title}\n\n{a.content[:1000]}" for a in articles]
    print(f"[index] Embedding {len(texts)} articles...")
    embeddings = embed_texts(texts)

    EMBEDDINGS_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDict(str(EMBEDDINGS_DB), autocommit=True) as db:
        for article, emb in zip(articles, embeddings):
            db[article.path] = emb.tobytes()
        db["__dim__"] = embeddings.shape[1]
        db["__count__"] = len(articles)

    print(f"[index] Embeddings stored: {len(articles)} articles, dim={embeddings.shape[1]}")


def embedding_search(query: str, top_k: int = 8) -> list[tuple[str, float]]:
    """Search with embedding cosine similarity."""
    if not EMBEDDINGS_DB.exists():
        return []

    q_emb = embed_texts([query])[0]

    paths = []
    vectors = []
    with SqliteDict(str(EMBEDDINGS_DB)) as db:
        dim = db.get("__dim__", 0)
        if not dim:
            return []
        for key, val in db.items():
            if key.startswith("__"):
                continue
            paths.append(key)
            vec = np.frombuffer(val, dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            vectors.append(vec)

    if not vectors:
        return []

    matrix = np.vstack(vectors)
    scores = matrix @ q_emb  # cosine sim (vectors are L2-normalized)
    top_indices = scores.argsort()[::-1][:top_k]
    return [(paths[i], float(scores[i])) for i in top_indices]


# ── Index rebuild entry point ─────────────────────────────────────────────────

def rebuild_index(use_embeddings: bool = True) -> dict:
    """Rebuild both TF-IDF and (optionally) embedding index."""
    articles = load_all_articles()
    if not articles:
        print("[index] No articles found.")
        return {"articles": 0}

    build_tfidf_index(articles)
    if use_embeddings:
        try:
            build_embedding_index(articles)
        except Exception as e:
            print(f"[warn] Embedding index failed (TF-IDF only): {e}")

    return {"articles": len(articles)}


def get_index_stats() -> dict:
    """Return stats about the current index state."""
    stats = {}
    if TFIDF_INDEX_PATH.exists():
        with open(TFIDF_INDEX_PATH, "rb") as f:
            _, matrix, paths = pickle.load(f)
        stats["tfidf_articles"] = len(paths)
        stats["tfidf_features"] = matrix.shape[1]
        import os
        stats["tfidf_mtime"] = os.path.getmtime(TFIDF_INDEX_PATH)
    if EMBEDDINGS_DB.exists():
        with SqliteDict(str(EMBEDDINGS_DB)) as db:
            stats["embedding_articles"] = db.get("__count__", 0)
            stats["embedding_dim"] = db.get("__dim__", 0)
    return stats
