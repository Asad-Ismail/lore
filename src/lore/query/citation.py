"""Extract and validate citations from wiki Q&A responses."""

from __future__ import annotations

import re
from pathlib import Path

from lore.config import WIKI_DIR


def extract_citations(response: str) -> list[str]:
    """Extract all [[ArticleName]] citation targets from a response."""
    return re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]", response)


def validate_citations(
    cited: list[str],
    retrieved_paths: list[str],
    wiki_dir: Path = WIKI_DIR,
) -> dict:
    """
    Validate citations for grounding and existence.

    Returns:
        {
            "valid": [...],        # exist in wiki AND were retrieved
            "hallucinated": [...], # cited but not in retrieved context
            "nonexistent": [...],  # cited but no wiki file found
        }
    """
    # Build set of existing article titles (normalized)
    existing_titles = {
        _normalize(f.stem.replace("-", " ").title())
        for f in wiki_dir.rglob("*.md")
        if not f.name.startswith("_")
    }

    # Build set of retrieved article titles
    retrieved_titles = {
        _normalize(Path(p).stem.replace("-", " ").title())
        for p in retrieved_paths
    }

    valid = []
    hallucinated = []
    nonexistent = []

    for cite in cited:
        n = _normalize(cite)
        if n not in existing_titles:
            nonexistent.append(cite)
        elif n not in retrieved_titles:
            hallucinated.append(cite)
        else:
            valid.append(cite)

    return {"valid": valid, "hallucinated": hallucinated, "nonexistent": nonexistent}


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()
