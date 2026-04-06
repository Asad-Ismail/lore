"""
FastAPI server for the evolving agent.

Two endpoints:
- POST /generate — inference using current LoRA model
- POST /train — trigger background training

Supports hot-swap of LoRA adapter without restarting.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from lore.config import LORA_CHECKPOINTS_DIR, LORA_BASE_MODEL_ID, HF_CACHE_DIR, get_device_map, get_torch_dtype

app = FastAPI(title="Wiki Evolving Agent", version="0.1.0")

# ── Model state ───────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_lock = threading.Lock()
_current_checkpoint: Path | None = None


def _load_model(checkpoint: Path | None = None):
    global _model, _tokenizer, _current_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

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
        print(f"[serve] Loading LoRA from: {checkpoint}")
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = base

    model.eval()
    _model = model
    _tokenizer = tokenizer
    _current_checkpoint = checkpoint
    print(f"[serve] Model ready")


def get_model():
    global _model, _tokenizer
    with _model_lock:
        if _model is None:
            latest = _get_latest_checkpoint()
            _load_model(latest)
        return _model, _tokenizer


def _get_latest_checkpoint() -> Path | None:
    if not LORA_CHECKPOINTS_DIR.exists():
        return None
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    return checkpoints[-1] if checkpoints else None


def _hot_swap_if_needed():
    """Check for new checkpoint signal and hot-swap if available."""
    global _model, _tokenizer, _current_checkpoint
    signal = LORA_CHECKPOINTS_DIR / ".new_checkpoint"
    if not signal.exists():
        return
    new_ckpt = Path(signal.read_text().strip())
    if new_ckpt == _current_checkpoint:
        return
    print(f"[serve] Hot-swapping to: {new_ckpt}")
    with _model_lock:
        _load_model(new_ckpt)
    signal.unlink(missing_ok=True)


# ── Background hot-swap watcher ───────────────────────────────────────────────

async def _watch_for_new_checkpoints():
    while True:
        await asyncio.sleep(30)
        _hot_swap_if_needed()


@app.on_event("startup")
async def startup():
    asyncio.create_task(_watch_for_new_checkpoints())


# ── API schemas ───────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    system: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    response: str
    model: str
    checkpoint: str | None


class TrainRequest(BaseModel):
    force: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model, tokenizer = get_model()

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return GenerateResponse(
        response=response,
        model=LORA_BASE_MODEL_ID,
        checkpoint=str(_current_checkpoint) if _current_checkpoint else None,
    )


@app.post("/train")
async def trigger_training(req: TrainRequest):
    """Trigger a background training run."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "lore.evolve.trainer", "--background"],
        start_new_session=True,
    )
    return {"status": "training_started", "pid": proc.pid}


@app.get("/status")
async def status():
    from lore.evolve.trajectory import get_trajectory_stats
    from lore.evolve.buffer import get_reward_stats

    traj_stats = get_trajectory_stats()
    reward_stats = get_reward_stats()
    return {
        "model": LORA_BASE_MODEL_ID,
        "checkpoint": str(_current_checkpoint) if _current_checkpoint else "base_model",
        "trajectories": traj_stats,
        "rewards": reward_stats,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
