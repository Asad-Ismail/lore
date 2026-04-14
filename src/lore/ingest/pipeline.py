"""Ingestion pipeline: fetch URLs, extract PDF text, deduplicate."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlitedict import SqliteDict

from lore.config import RAW_DIR, FINGERPRINTS_DB
from lore.ingest.parsers import parse_file, RawDocument


@dataclass
class IngestedSource:
    """Normalized ingest result for files and URLs."""

    source_ref: str
    raw_path: str
    title: str
    content: str
    source_type: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _store_fingerprint_record(fingerprint: str, path: Path, doc: RawDocument) -> None:
    with SqliteDict(str(FINGERPRINTS_DB), autocommit=True) as fp_db:
        fp_db[fingerprint] = {
            "path": str(path),
            "title": doc.title,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "extracted_text": doc.content[:50000],
        }


def ingest_file_result(path: str | Path, force: bool = False) -> IngestedSource:
    """
    Parse a file and register its fingerprint for dedup.
    Returns a normalized ingest result with title, path, and extracted text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    doc = parse_file(path)
    if doc is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    fingerprint = file_sha256(path)
    FINGERPRINTS_DB.parent.mkdir(parents=True, exist_ok=True)

    with SqliteDict(str(FINGERPRINTS_DB), autocommit=True) as fp_db:
        if not force and fingerprint in fp_db:
            cached = fp_db[fingerprint]
            print(f"[skip] Already ingested: {path.name} ({fingerprint[:8]})")
            return IngestedSource(
                source_ref=str(path),
                raw_path=str(path),
                title=cached.get("title", doc.title),
                content=cached.get("extracted_text", doc.content[:50000]),
                source_type=doc.source_type,
            )

    _store_fingerprint_record(fingerprint, path, doc)
    print(f"[ok] Ingested: {path.name} ({len(doc.content)} chars)")
    return IngestedSource(
        source_ref=str(path),
        raw_path=str(path),
        title=doc.title,
        content=doc.content,
        source_type=doc.source_type,
    )


def ingest_file(path: str | Path, force: bool = False) -> str:
    """
    Parse a file and register its fingerprint for dedup.
    Returns the extracted text content (for the agent to read).
    """
    return ingest_file_result(path, force=force).content


def _is_arxiv_url(url: str) -> bool:
    return "arxiv.org" in url


def _arxiv_pdf_url(url: str) -> str | None:
    """Extract arXiv PDF URL from any arXiv link (/abs/, /html/, /pdf/)."""
    m = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d+\.\d+)(v\d+)?", url)
    if m:
        paper_id = m.group(1)
        version = m.group(2) or ""
        return f"https://arxiv.org/pdf/{paper_id}{version}"
    return None


def _slugify_url(url: str) -> str:
    return re.sub(r"[^\w]+", "-", url.split("//")[-1].rstrip("/"))[:80]


def _classify_url_subdir(url: str) -> str:
    paper_domains = ["arxiv.org", "openreview.net", "aclanthology.org", "semanticscholar.org"]
    return "papers" if any(d in url for d in paper_domains) else "articles"


def _download_pdf(url: str, dest: Path) -> None:
    import httpx
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def ingest_url_result(url: str) -> IngestedSource:
    """
    Fetch a URL and save to raw/.
    For arXiv: downloads the actual PDF.
    For web articles: fetches HTML and converts to markdown.
    Returns a normalized ingest result with title, saved raw path, and text.
    """
    slug = _slugify_url(url)
    subdir = _classify_url_subdir(url)

    pdf_url = _arxiv_pdf_url(url) if _is_arxiv_url(url) else None
    if pdf_url:
        dest = RAW_DIR / subdir / f"{slug}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] Downloading PDF: {pdf_url}")
        _download_pdf(pdf_url, dest)
        print(f"[ok] Saved to {dest}")
        result = ingest_file_result(dest)
        result.source_ref = url
        return result

    import httpx
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    try:
        import markdownify
        content = markdownify.markdownify(html, heading_style="ATX")
    except ImportError:
        content = re.sub(r"<[^>]+>", " ", html)
        content = re.sub(r"\s+", " ", content).strip()

    dest = RAW_DIR / subdir / f"{slug}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)

    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_m.group(1).strip() if title_m else slug

    dest.write_text(f"# {title}\n\nSource: {url}\n\n{content}", encoding="utf-8")
    print(f"[ok] Saved to {dest}")
    result = ingest_file_result(dest)
    result.source_ref = url
    return result


def ingest_url(url: str) -> str:
    """
    Fetch a URL and save to raw/.
    For arXiv: downloads the actual PDF.
    For web articles: fetches HTML and converts to markdown.
    Returns extracted text content.
    """
    return ingest_url_result(url).content


def get_ingestion_stats() -> dict:
    """Return stats about ingested sources."""
    if not FINGERPRINTS_DB.exists():
        return {"sources": 0}
    with SqliteDict(str(FINGERPRINTS_DB)) as db:
        return {"sources": len(db)}
