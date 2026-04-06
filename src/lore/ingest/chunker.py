"""Semantic chunking: split text into token-budget chunks with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from lore.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


@dataclass
class TextChunk:
    content: str
    position: int        # chunk index within parent
    char_start: int
    char_end: int
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    """Fast token estimate: ~0.75 tokens per word, good enough for chunking."""
    return int(len(text.split()) * 0.75)


def _split_paragraphs(text: str) -> list[str]:
    """Split on double newlines, preserving paragraph structure."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks that respect paragraph boundaries.

    Strategy:
    1. Split into paragraphs
    2. Greedily accumulate paragraphs up to chunk_size tokens
    3. When budget exhausted, emit chunk and backtrack by `overlap` tokens
    """
    if not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    char_pos = 0
    chunk_idx = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If a single paragraph exceeds chunk_size, hard-split it by sentences
        if para_tokens > chunk_size:
            for sub in _split_large_paragraph(para, chunk_size):
                sub_tokens = _estimate_tokens(sub)
                if current_tokens + sub_tokens > chunk_size and current_paras:
                    content = "\n\n".join(current_paras)
                    chunks.append(TextChunk(
                        content=content,
                        position=chunk_idx,
                        char_start=char_pos,
                        char_end=char_pos + len(content),
                        token_estimate=current_tokens,
                    ))
                    chunk_idx += 1
                    char_pos += len(content)
                    # Overlap: keep last N tokens worth of paragraphs
                    current_paras, current_tokens = _trim_to_overlap(current_paras, overlap)
                current_paras.append(sub)
                current_tokens += sub_tokens
            continue

        if current_tokens + para_tokens > chunk_size and current_paras:
            content = "\n\n".join(current_paras)
            chunks.append(TextChunk(
                content=content,
                position=chunk_idx,
                char_start=char_pos,
                char_end=char_pos + len(content),
                token_estimate=current_tokens,
            ))
            chunk_idx += 1
            char_pos += len(content)
            current_paras, current_tokens = _trim_to_overlap(current_paras, overlap)

        current_paras.append(para)
        current_tokens += para_tokens

    # Emit final chunk
    if current_paras:
        content = "\n\n".join(current_paras)
        chunks.append(TextChunk(
            content=content,
            position=chunk_idx,
            char_start=char_pos,
            char_end=char_pos + len(content),
            token_estimate=current_tokens,
        ))

    return chunks


def _split_large_paragraph(text: str, max_tokens: int) -> list[str]:
    """Split a large paragraph into sentence-based sub-chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    subs: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = _estimate_tokens(sent)
        if current_tokens + t > max_tokens and current:
            subs.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += t
    if current:
        subs.append(" ".join(current))
    return subs


def _trim_to_overlap(paras: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Keep only the trailing paragraphs that fit within overlap_tokens."""
    result: list[str] = []
    total = 0
    for para in reversed(paras):
        t = _estimate_tokens(para)
        if total + t > overlap_tokens:
            break
        result.insert(0, para)
        total += t
    return result, total
