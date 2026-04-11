"""Parsers for different raw source document types."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class RawDocument:
    """Normalized representation of any source document."""
    content: str
    title: str
    source_path: str
    source_type: str  # pdf | markdown | text | csv | image | json
    metadata: dict = field(default_factory=dict)


class Parser(Protocol):
    def can_parse(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> RawDocument: ...


# ── Markdown parser ───────────────────────────────────────────────────────────

class MarkdownParser:
    EXTENSIONS = {".md", ".markdown"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _extract_md_title(text) or path.stem
        metadata = _extract_frontmatter(text)
        # Strip frontmatter from content
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL).strip()
        return RawDocument(
            content=content,
            title=title,
            source_path=str(path),
            source_type="markdown",
            metadata=metadata,
        )


def _extract_md_title(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _extract_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}
    try:
        import yaml
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


# ── Plain text parser ─────────────────────────────────────────────────────────

class TextParser:
    EXTENSIONS = {".txt", ".rst"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = path.stem.replace("_", " ").replace("-", " ").title()
        return RawDocument(
            content=text,
            title=title,
            source_path=str(path),
            source_type="text",
        )


# ── PDF parser ────────────────────────────────────────────────────────────────

class PDFParser:
    EXTENSIONS = {".pdf"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> RawDocument:
        text = self._extract_text(path)
        title = _guess_pdf_title(text) or path.stem
        return RawDocument(
            content=text,
            title=title,
            source_path=str(path),
            source_type="pdf",
        )

    def _extract_text(self, path: Path) -> str:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(path))
        except ImportError:
            pass
        # Fallback: try pdftotext CLI
        import subprocess
        try:
            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return f"[PDF could not be parsed: {path.name}]"


def _guess_pdf_title(text: str) -> str | None:
    """Heuristic: first non-empty line that looks like a title."""
    for line in text.splitlines()[:20]:
        line = line.strip()
        if 10 < len(line) < 150 and not line.endswith("."):
            return line
    return None


# ── CSV parser ────────────────────────────────────────────────────────────────

class CSVParser:
    EXTENSIONS = {".csv", ".tsv"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> RawDocument:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return RawDocument(
                content="[Empty CSV]",
                title=path.stem,
                source_path=str(path),
                source_type="csv",
            )
        # Convert rows to readable markdown table
        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows[:500]:  # Cap at 500 rows
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return RawDocument(
            content="\n".join(lines),
            title=path.stem.replace("_", " ").title(),
            source_path=str(path),
            source_type="csv",
            metadata={"row_count": len(rows), "columns": headers},
        )


# ── JSON export parser (Day One, Apple Notes, etc.) ───────────────────────────

class JSONExportParser:
    EXTENSIONS = {".json"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def parse(self, path: Path) -> RawDocument:
        data = json.loads(path.read_text(encoding="utf-8"))
        content = self._extract_content(data)
        title = path.stem.replace("_", " ").title()
        return RawDocument(
            content=content,
            title=title,
            source_path=str(path),
            source_type="json",
        )

    def _extract_content(self, data) -> str:
        # Day One format: {"entries": [{"text": ..., "creationDate": ...}]}
        if isinstance(data, dict) and "entries" in data:
            parts = []
            for entry in data["entries"]:
                date = entry.get("creationDate", "")
                text = entry.get("text", "")
                if text:
                    parts.append(f"### {date}\n\n{text}")
            return "\n\n---\n\n".join(parts)
        # Generic: dump as pretty-printed text
        return json.dumps(data, indent=2)


# ── Registry ──────────────────────────────────────────────────────────────────

_PARSERS: list[Parser] = [
    MarkdownParser(),
    TextParser(),
    PDFParser(),
    CSVParser(),
    JSONExportParser(),
]


def get_parser(path: Path) -> Parser | None:
    for parser in _PARSERS:
        if parser.can_parse(path):
            return parser
    return None


def parse_file(path: Path) -> RawDocument | None:
    parser = get_parser(path)
    if parser is None:
        return None
    return parser.parse(path)
