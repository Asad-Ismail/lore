"""Track which wiki articles need recompilation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlitedict import SqliteDict
from lore.config import DATA_DIR, WIKI_DIR


_COMPILE_STATE_DB = DATA_DIR / "compile_state.db"


def get_compiled_source_hashes() -> dict[str, str]:
    """Return map of wiki_article_path → source_chunk_hash."""
    if not _COMPILE_STATE_DB.exists():
        return {}
    with SqliteDict(str(_COMPILE_STATE_DB)) as db:
        return dict(db)


def record_compilation(wiki_path: str, chunk_ids: list[str], combined_hash: str) -> None:
    """Record that a wiki article was compiled from these chunks."""
    with SqliteDict(str(_COMPILE_STATE_DB), autocommit=True) as db:
        db[wiki_path] = {
            "chunk_ids": chunk_ids,
            "combined_hash": combined_hash,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
        }


def needs_recompile(wiki_path: str, current_chunk_hash: str) -> bool:
    """Return True if the wiki article needs recompilation."""
    if not _COMPILE_STATE_DB.exists():
        return True
    with SqliteDict(str(_COMPILE_STATE_DB)) as db:
        state = db.get(wiki_path)
        if state is None:
            return True
        return state.get("combined_hash") != current_chunk_hash


def compute_combined_hash(chunk_ids: list[str]) -> str:
    """Hash a sorted list of chunk IDs to detect when source material changed."""
    import hashlib
    content = "|".join(sorted(chunk_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]
