"""
On-Policy Distillation (OPD) bootstrap mode.

When reward variance is too low for GRPO to get a useful signal
(early stage, or after a distribution shift), we use Qwen3-4B (teacher)
to generate high-quality reference answers and train Qwen3-1.7B (student)
via SFT cross-entropy loss.

Switch criteria: use OPD when std(rewards_in_buffer) < OPD_SWITCH_STD
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lore.config import (
    INFERENCE_MODEL_ID, LORA_BASE_MODEL_ID, HF_CACHE_DIR,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    LEARNING_RATE, MAX_GRAD_NORM, LORA_CHECKPOINTS_DIR,
    OPD_BOOTSTRAP_N, OPD_SWITCH_STD,
)
from lore.evolve.buffer import sample_trajectories, get_reward_stats, mark_trajectories_trained
from lore.evolve.trajectory import Trajectory


def should_use_opd() -> bool:
    """Return True if we should use OPD instead of GRPO."""
    stats = get_reward_stats()
    total = stats.get("total", 0)
    std = stats.get("recent_std", 0.0)

    if total < OPD_BOOTSTRAP_N:
        return True  # Always use OPD for first N trajectories
    if std < OPD_SWITCH_STD:
        return True  # Low variance: GRPO would get no signal
    return False


def run_opd_step(
    student_model,
    student_tokenizer,
    trajectories: list[Trajectory],
    optimizer,
    max_steps: int = 10,
) -> float:
    """
    Run OPD training: generate teacher answers → train student via SFT.
    Returns mean training loss.
    """
    print(f"[OPD] Running distillation on {len(trajectories)} trajectories")

    # Generate teacher answers
    from lore.compile.compiler import generate as teacher_generate

    losses = []
    for traj in trajectories[:max_steps]:
        # Generate a high-quality teacher answer
        teacher_answer = teacher_generate(
            f"Wiki context:\n{traj.context[:2000]}\n\nQuestion: {traj.question}\nAnswer:",
        )

        # Format as training pair
        prompt = f"Question: {traj.question}\n\nContext: {traj.context[:1500]}\n\nAnswer:"
        full_text = prompt + " " + teacher_answer

        inputs = student_tokenizer(
            full_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(student_model.device)

        # SFT: cross-entropy on the answer tokens only
        labels = inputs["input_ids"].clone()
        prompt_len = len(student_tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        labels[0, :prompt_len] = -100  # Mask prompt tokens

        optimizer.zero_grad()
        outputs = student_model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    print(f"[OPD] Mean SFT loss: {mean_loss:.4f}")
    return mean_loss
