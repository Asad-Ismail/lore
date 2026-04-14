"""Helpers for deriving display titles from file names without mangling acronyms."""

from __future__ import annotations

from pathlib import Path


def stem_to_title(stem: str) -> str:
    """Convert a filename stem to a human-readable title while preserving casing."""
    return stem.replace("_", " ").replace("-", " ").strip()


def path_to_title(path: Path) -> str:
    """Convert a path to a display title using its stem."""
    return stem_to_title(path.stem)
