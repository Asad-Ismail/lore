"""
Embedding utilities for the wiki index.

Thin module providing the embedding interface used by store.py and search.py.
Actual model loading and batched inference live here so store.py stays focused
on storage concerns.
"""

from __future__ import annotations

import numpy as np
import torch

from lore.config import LORA_BASE_MODEL_ID, HF_CACHE_DIR

_model_cache = None


def get_embedding_model():
    """Lazy-load Qwen3-1.7B for mean-pool embeddings. Cached for the process lifetime."""
    global _model_cache
    if _model_cache is None:
        from transformers import AutoModel, AutoTokenizer
        print(f"[embedder] Loading: {LORA_BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(
            LORA_BASE_MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            LORA_BASE_MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _model_cache = (model, tokenizer)
    return _model_cache


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Embed a list of texts using mean-pooling over last hidden states.
    Returns L2-normalised float32 array of shape (len(texts), hidden_dim).
    """
    model, tokenizer = get_embedding_model()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state          # (B, T, D)
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, D)

        vecs = pooled.float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        all_embeddings.append(vecs / np.clip(norms, 1e-9, None))

    return np.vstack(all_embeddings)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns 1-D normalised float32 array."""
    return embed([query])[0]
