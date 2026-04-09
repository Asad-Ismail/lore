"""LLM-driven wiki article synthesis from raw chunks."""

from __future__ import annotations

import re
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lore.config import (
    WIKI_DIR, DATA_DIR, ABSORB_PENDING_FLAG,
    COMPILE_SYSTEM_PROMPT, INFERENCE_MODEL_ID, HF_CACHE_DIR,
    MAX_NEW_TOKENS, TEMPERATURE, get_device_map, get_torch_dtype,
)
from lore.ingest.pipeline import get_unabsorbed_chunks, mark_chunks_absorbed, IngestedChunk
from lore.compile.taxonomy import classify_article, article_path
from lore.compile.linker import rebuild_all_backlinks, snap_wikilinks
from lore.compile.incremental import record_compilation, compute_combined_hash, needs_recompile


def load_inference_model():
    """Load Qwen3-4B for wiki compilation. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading inference model: {INFERENCE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        INFERENCE_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        INFERENCE_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        torch_dtype=get_torch_dtype(),
        device_map=get_device_map(),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


_model_cache: tuple | None = None


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_inference_model()
    return _model_cache


def generate(prompt: str, system: str = COMPILE_SYSTEM_PROMPT) -> str:
    """Run inference with Qwen3-4B."""
    model, tokenizer = get_model()
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3 thinking mode off for compilation
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip input tokens from output
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_concepts(chunks: list[IngestedChunk]) -> list[str]:
    """
    Ask the LLM to identify key concepts from a batch of chunks.
    Returns a deduplicated list of concept names.
    """
    combined = "\n\n---\n\n".join(
        f"[Source: {c.title}]\n{c.content[:600]}"
        for c in chunks[:20]  # Limit context
    )
    prompt = (
        "Given these ML research excerpts, list the key concepts, techniques, models, "
        "papers, or people that each deserve their own wiki article.\n\n"
        "Output ONLY a JSON array of strings, e.g.: [\"LoRA\", \"GPTQ\", \"Llama-3\"]\n\n"
        f"Excerpts:\n{combined}"
    )
    response = generate(prompt, system="You are a knowledge extraction assistant. Output only valid JSON.")
    # Extract JSON array from response
    m = re.search(r"\[.*?\]", response, re.DOTALL)
    if not m:
        # Fallback: extract quoted strings
        return re.findall(r'"([^"]+)"', response)
    try:
        import json
        concepts = json.loads(m.group())
        return [str(c).strip() for c in concepts if c]
    except Exception:
        return re.findall(r'"([^"]+)"', response)


def synthesize_article(concept: str, chunks: list[IngestedChunk]) -> tuple[str, list[str]]:
    """
    Compile a wiki article for `concept` from the given source chunks.
    Returns (article_content, source_paths).
    """
    source_text = "\n\n---\n\n".join(
        f"[Source: {c.title}, chunk {c.position}]\n{c.content}"
        for c in chunks[:12]
    )
    source_paths = list({c.source_path for c in chunks})

    prompt = (
        f"Source excerpts about '{concept}':\n\n{source_text}\n\n"
        f"Write the wiki article for: **{concept}**"
    )
    return generate(prompt), source_paths


def write_article(concept: str, content: str, category: str, source_paths: list[str] | None = None) -> Path:
    """Write a wiki article to disk. Returns the path written."""
    rel_path = article_path(category, concept)
    full_path = WIKI_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    content = snap_wikilinks(content)

    if not content.startswith("---"):
        now = datetime.now(timezone.utc).isoformat()
        sources_yaml = ""
        if source_paths:
            sources_yaml = "sources:\n" + "".join(f"  - {p}\n" for p in source_paths)
        frontmatter = (
            f"---\n"
            f"title: {concept}\n"
            f"category: {category}\n"
            f"created: {now}\n"
            f"updated: {now}\n"
            f"{sources_yaml}"
            f"---\n\n"
        )
        content = frontmatter + content

    full_path.write_text(content, encoding="utf-8")
    return full_path


def absorb(force: bool = False) -> dict:
    """
    Main compilation loop: process unabsorbed chunks into wiki articles.
    Returns stats dict.
    """
    chunks = get_unabsorbed_chunks()
    if not chunks:
        print("[absorb] Nothing to absorb.")
        return {"new_articles": 0, "updated_articles": 0, "chunks_processed": 0}

    print(f"[absorb] Processing {len(chunks)} unabsorbed chunks...")

    # Group chunks by concept
    concepts = extract_concepts(chunks)
    print(f"[absorb] Identified {len(concepts)} concepts: {concepts[:10]}")

    # For each concept, find relevant chunks
    concept_chunks: dict[str, list[IngestedChunk]] = defaultdict(list)
    for concept in concepts:
        concept_lower = concept.lower()
        for chunk in chunks:
            if concept_lower in chunk.content.lower() or concept_lower in chunk.title.lower():
                concept_chunks[concept].append(chunk)

    # Synthesize articles
    new_articles = 0
    updated_articles = 0
    all_absorbed_chunk_ids: list[str] = []

    for concept, rel_chunks in concept_chunks.items():
        if not rel_chunks:
            continue

        chunk_ids = [c.chunk_id for c in rel_chunks]
        combined_hash = compute_combined_hash(chunk_ids)

        # Preliminary classification (no content yet) — only used for the
        # incremental-recompile check, which needs a stable path key.
        prelim_rel_path = article_path(classify_article(concept), concept)

        if not force and not needs_recompile(prelim_rel_path, combined_hash):
            print(f"[skip] Up to date: {concept}")
            continue

        print(f"[compile] {concept} ({len(rel_chunks)} chunks)...")
        article_content, source_paths = synthesize_article(concept, rel_chunks)

        # Accurate classification with content snippet; used for the actual write.
        category = classify_article(concept, article_content[:300])
        final_rel_path = article_path(category, concept)

        # Check is_new BEFORE writing so we know whether the file pre-existed.
        is_new = not (WIKI_DIR / final_rel_path).exists()

        path = write_article(concept, article_content, category, source_paths=source_paths)

        if is_new:
            new_articles += 1
        else:
            updated_articles += 1

        # Record against the final (content-accurate) path so incremental
        # checks on subsequent absorb runs resolve to the right file.
        record_compilation(final_rel_path, chunk_ids, combined_hash)
        all_absorbed_chunk_ids.extend(chunk_ids)
        print(f"[ok] {'New' if is_new else 'Updated'}: {path.relative_to(WIKI_DIR)}")

    # Mark all processed chunks as absorbed
    if all_absorbed_chunk_ids:
        mark_chunks_absorbed(list(set(all_absorbed_chunk_ids)))

    # Rebuild backlinks
    print("[absorb] Rebuilding backlinks...")
    rebuild_all_backlinks()

    # Set flag for index rebuild hook
    ABSORB_PENDING_FLAG.parent.mkdir(parents=True, exist_ok=True)
    ABSORB_PENDING_FLAG.touch()

    stats = {
        "new_articles": new_articles,
        "updated_articles": updated_articles,
        "chunks_processed": len(chunks),
        "concepts_found": len(concepts),
    }
    print(f"[absorb] Done: {stats}")
    return stats
