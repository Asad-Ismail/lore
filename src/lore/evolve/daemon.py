"""
Daemon process: keeps the LoRA model in memory for instant suggestions.

Run once in a terminal:
    uv run lore-train serve

Then all suggestion/training calls go through HTTP (~100ms) instead of
loading the model from scratch (~17s) each time.

Endpoints:
    GET  /suggest?n=3     — generate follow-up questions
    POST /train/curiosity — trigger curiosity training
    GET  /status          — model + training stats
    GET  /health          — liveness check
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from lore.config import LORA_CHECKPOINTS_DIR, LORA_BASE_MODEL_ID, HF_CACHE_DIR, get_device_map, get_torch_dtype

app = FastAPI(title="Lore Daemon", version="0.1.0")

_model = None
_tokenizer = None
_lock = threading.Lock()
_current_checkpoint: Path | None = None


def _get_latest_checkpoint() -> Path | None:
    if not LORA_CHECKPOINTS_DIR.exists():
        return None
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    return checkpoints[-1] if checkpoints else None


def _load_model(checkpoint: Path | None = None):
    global _model, _tokenizer, _current_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[daemon] Loading model: {LORA_BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_BASE_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        torch_dtype=get_torch_dtype(),
        device_map=get_device_map(),
        trust_remote_code=True,
    )

    if checkpoint and checkpoint.exists():
        print(f"[daemon] Loading LoRA checkpoint: {checkpoint.name}")
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = base

    model.eval()
    _model = model
    _tokenizer = tokenizer
    _current_checkpoint = checkpoint
    print(f"[daemon] Model ready (checkpoint: {checkpoint.name if checkpoint else 'base'})")


def _ensure_model():
    global _model
    with _lock:
        if _model is None:
            _load_model(_get_latest_checkpoint())
        return _model, _tokenizer


def _reload_if_new_checkpoint():
    """Check for new checkpoint and reload if needed."""
    global _current_checkpoint
    latest = _get_latest_checkpoint()
    if latest and latest != _current_checkpoint:
        print(f"[daemon] New checkpoint detected: {latest.name}")
        with _lock:
            _load_model(latest)


@app.on_event("startup")
async def startup():
    _ensure_model()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/status")
async def status():
    from lore.evolve.trajectory import get_question_trace_stats
    return {
        "model": LORA_BASE_MODEL_ID,
        "checkpoint": _current_checkpoint.name if _current_checkpoint else "base",
        "question_traces": get_question_trace_stats(),
    }


class SuggestResponse(BaseModel):
    suggestions: list[dict]


@app.get("/suggest", response_model=SuggestResponse)
async def suggest(n: int = 3):
    _reload_if_new_checkpoint()

    from lore.evolve.curiosity import (
        build_wiki_state_summary, question_reward, CURIOSITY_SYSTEM_PROMPT,
    )
    from lore.evolve.trajectory import get_all_past_questions

    model, tokenizer = _ensure_model()
    wiki_state = build_wiki_state_summary()
    past_questions = get_all_past_questions()

    if wiki_state == "[Empty wiki]":
        return SuggestResponse(suggestions=[])

    prompt = (
        f"Wiki state:\n{wiki_state}\n\n"
        f"Recent questions asked:\n"
        + "\n".join(f"- {q}" for q in past_questions[-10:])
        + "\n\nGenerate a follow-up question this researcher should explore:"
    )

    suggestions = []
    for _ in range(n * 2):
        messages = [
            {"role": "system", "content": CURIOSITY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        candidate = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not candidate or len(candidate) < 10:
            continue
        candidate = candidate.split("\n")[0].strip()

        reward = question_reward(candidate, wiki_state, past_questions)
        suggestions.append({"question": candidate, **reward})

    suggestions.sort(key=lambda s: s["combined"], reverse=True)

    import re
    def _norm(s): return re.sub(r"[^\w\s]", "", s.lower()).strip()
    seen = set()
    unique = []
    for s in suggestions:
        q_norm = _norm(s["question"])
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(s)
        if len(unique) >= n:
            break

    return SuggestResponse(suggestions=unique)


@app.post("/train/curiosity")
async def train_curiosity():
    """Trigger curiosity training in a background thread."""
    def _run():
        _reload_if_new_checkpoint()
        from lore.evolve.trainer import run_curiosity_training
        run_curiosity_training()
        _reload_if_new_checkpoint()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "training_started"}
