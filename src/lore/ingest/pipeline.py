"""Ingestion pipeline: parse → fingerprint → chunk → store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlitedict import SqliteDict

from lore.config import (
    RAW_DIR, DATA_DIR, FINGERPRINTS_DB,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
)
from lore.ingest.parsers import parse_file, RawDocument
from lore.ingest.chunker import chunk_text, TextChunk


@dataclass
class IngestedChunk:
    chunk_id: str           # SHA-256 of content
    source_path: str
    source_type: str
    title: str
    position: int
    content: str
    token_estimate: int
    absorbed: bool = False
    metadata: dict = field(default_factory=dict)
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: str | Path, force: bool = False) -> list[IngestedChunk]:
    """
    Parse a file, fingerprint it, chunk it, and store chunks.
    Returns list of IngestedChunk objects created (empty if already ingested).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    fingerprint = file_sha256(path)

    with SqliteDict(str(FINGERPRINTS_DB), autocommit=True) as fp_db:
        if not force and fingerprint in fp_db:
            print(f"[skip] Already ingested: {path.name} ({fingerprint[:8]})")
            return []

    doc = parse_file(path)
    if doc is None:
        print(f"[skip] Unsupported file type: {path.suffix}")
        return []

    chunks = chunk_text(doc.content)
    if not chunks:
        print(f"[skip] No content extracted from: {path.name}")
        return []

    ingested: list[IngestedChunk] = []

    with SqliteDict(str(FINGERPRINTS_DB), autocommit=True) as fp_db:
        chunks_db_path = DATA_DIR / "chunks.db"
        with SqliteDict(str(chunks_db_path), autocommit=True) as chunks_db:
            for chunk in chunks:
                chunk_id = sha256(chunk.content)
                ic = IngestedChunk(
                    chunk_id=chunk_id,
                    source_path=str(path),
                    source_type=doc.source_type,
                    title=doc.title,
                    position=chunk.position,
                    content=chunk.content,
                    token_estimate=chunk.token_estimate,
                    absorbed=False,
                    metadata={**doc.metadata, "file_sha256": fingerprint},
                )
                chunks_db[chunk_id] = asdict(ic)
                ingested.append(ic)

        fp_db[fingerprint] = {
            "path": str(path),
            "title": doc.title,
            "chunk_count": len(ingested),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    print(f"[ok] Ingested: {path.name} → {len(ingested)} chunks")
    return ingested


def ingest_url(url: str) -> list[IngestedChunk]:
    """
    Fetch a URL and ingest it as a markdown document.
    Saves to raw/articles/<slug>.md, then ingests that file.
    """
    import re
    import subprocess
    import sys

    # Use httpx for fetching
    try:
        import httpx
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    # Convert HTML to markdown using markdownify if available, else strip tags
    try:
        import markdownify
        content = markdownify.markdownify(html, heading_style="ATX")
    except ImportError:
        # Fallback: strip HTML tags
        content = re.sub(r"<[^>]+>", " ", html)
        content = re.sub(r"\s+", " ", content).strip()

    # Generate slug from URL
    slug = re.sub(r"[^\w]+", "-", url.split("//")[-1].rstrip("/"))[:80]
    dest = RAW_DIR / "articles" / f"{slug}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Extract title from HTML
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_m.group(1).strip() if title_m else slug

    dest.write_text(f"# {title}\n\nSource: {url}\n\n{content}", encoding="utf-8")
    print(f"[ok] Saved to {dest}")
    return ingest_file(dest)


def get_unabsorbed_chunks() -> list[IngestedChunk]:
    """Return all chunks not yet compiled into wiki articles."""
    chunks_db_path = DATA_DIR / "chunks.db"
    if not chunks_db_path.exists():
        return []
    result = []
    with SqliteDict(str(chunks_db_path)) as db:
        for chunk_id, data in db.items():
            if not data.get("absorbed", False):
                result.append(IngestedChunk(**data))
    return result


def mark_chunks_absorbed(chunk_ids: list[str]) -> None:
    """Mark chunks as absorbed after wiki compilation."""
    chunks_db_path = DATA_DIR / "chunks.db"
    with SqliteDict(str(chunks_db_path), autocommit=True) as db:
        for chunk_id in chunk_ids:
            if chunk_id in db:
                entry = dict(db[chunk_id])
                entry["absorbed"] = True
                db[chunk_id] = entry


def get_ingestion_stats() -> dict:
    """Return stats about the ingestion state."""
    chunks_db_path = DATA_DIR / "chunks.db"
    if not chunks_db_path.exists():
        return {"total_chunks": 0, "absorbed": 0, "unabsorbed": 0, "sources": 0}

    total = absorbed = 0
    sources: set[str] = set()
    with SqliteDict(str(chunks_db_path)) as db:
        for data in db.values():
            total += 1
            if data.get("absorbed"):
                absorbed += 1
            sources.add(data.get("source_path", ""))

    return {
        "total_chunks": total,
        "absorbed": absorbed,
        "unabsorbed": total - absorbed,
        "sources": len(sources),
    }
