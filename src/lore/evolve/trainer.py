"""
GRPO LoRA training loop for the evolving agent.

Architecture:
- Base model: Qwen/Qwen3-1.7B (~4 GB VRAM in bf16)
- LoRA adapter: r=16, alpha=32, targets q/k/v/o_proj (~50 MB)
- GRPO: generate G=4 responses per prompt, use relative rewards as advantages
- Async: runs as background subprocess, hot-swaps adapter in serve.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lore.config import (
    LORA_BASE_MODEL_ID, HF_CACHE_DIR, LORA_CHECKPOINTS_DIR,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    GRPO_GROUP_SIZE, GRPO_BATCH_SIZE, TRAIN_BUFFER_SAMPLE,
    LEARNING_RATE, MAX_GRAD_NORM, get_device_map, get_torch_dtype,
)
from lore.evolve.buffer import (
    sample_trajectories, mark_trajectories_trained, get_reward_stats,
)
from lore.evolve.reward import compute_full_reward
from lore.evolve.distill import should_use_opd, run_opd_step
from lore.evolve.trajectory import Trajectory


def load_student_model():
    """Load Qwen3-1.7B with LoRA adapter."""
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_BASE_MODEL_ID,
        cache_dir=str(HF_CACHE_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for existing checkpoint
    latest_ckpt = _get_latest_checkpoint()
    if latest_ckpt:
        print(f"[trainer] Loading LoRA checkpoint: {latest_ckpt}")
        base = AutoModelForCausalLM.from_pretrained(
            LORA_BASE_MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
            torch_dtype=get_torch_dtype(),
            device_map=get_device_map(),
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(latest_ckpt))
    else:
        print(f"[trainer] Loading base model: {LORA_BASE_MODEL_ID}")
        base = AutoModelForCausalLM.from_pretrained(
            LORA_BASE_MODEL_ID,
            cache_dir=str(HF_CACHE_DIR),
            torch_dtype=get_torch_dtype(),
            device_map=get_device_map(),
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(base, lora_config)
        model.print_trainable_parameters()

    model.train()
    return model, tokenizer


def _get_latest_checkpoint() -> Path | None:
    """Return the most recent LoRA checkpoint directory, if any."""
    LORA_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    return checkpoints[-1] if checkpoints else None


def generate_group_responses(
    model,
    tokenizer,
    prompt: str,
    group_size: int = GRPO_GROUP_SIZE,
    max_new_tokens: int = 256,
) -> list[tuple[str, torch.Tensor]]:
    """
    Generate G responses for a prompt.
    Returns list of (response_text, log_probs_tensor).
    """
    responses = []
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    for _ in range(group_size):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = output.sequences[0][input_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute per-token log probs
        log_probs = torch.stack(output.scores, dim=1)  # (1, seq, vocab)
        log_probs = F.log_softmax(log_probs[0], dim=-1)  # (seq, vocab)
        token_log_probs = log_probs.gather(
            1, generated_ids.unsqueeze(1)
        ).squeeze(1)  # (seq,)

        responses.append((response_text, token_log_probs))

    return responses


def grpo_loss(
    model,
    tokenizer,
    trajectory: Trajectory,
    responses_with_rewards: list[tuple[str, torch.Tensor, float]],
) -> torch.Tensor:
    """
    Compute GRPO loss for a set of group responses.

    Loss = -mean( A_i * mean(log_prob_i) )
    where A_i = normalized advantage = (r_i - mean(r)) / (std(r) + eps)
    """
    rewards = torch.tensor([r for _, _, r in responses_with_rewards])

    # Group-relative advantage normalization
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = torch.tensor(0.0, requires_grad=True).to(model.device)
    count = 0

    for (response_text, log_probs, _), advantage in zip(responses_with_rewards, advantages):
        if log_probs.numel() == 0:
            continue
        mean_log_prob = log_probs.mean()
        total_loss = total_loss - advantage.to(model.device) * mean_log_prob
        count += 1

    return total_loss / max(count, 1)


def run_training(background: bool = False) -> dict:
    """
    Main training loop. Returns stats dict.
    Runs OPD if reward variance is low, GRPO otherwise.
    """
    print(f"[trainer] Starting training run at {datetime.now(timezone.utc).isoformat()}")

    trajectories = sample_trajectories(n=TRAIN_BUFFER_SAMPLE, strategy="curriculum")
    if not trajectories:
        print("[trainer] No trajectories to train on.")
        return {"steps": 0, "reason": "no_trajectories"}

    use_opd = should_use_opd()
    print(f"[trainer] Mode: {'OPD (distillation)' if use_opd else 'GRPO (RL)'}")
    print(f"[trainer] Training on {len(trajectories)} trajectories")

    # Load student model
    model, tokenizer = load_student_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    stats = {"steps": 0, "mean_loss": 0.0, "mode": "opd" if use_opd else "grpo"}
    losses = []

    if use_opd:
        # OPD: distillation from teacher
        mean_loss = run_opd_step(model, tokenizer, trajectories, optimizer)
        losses.append(mean_loss)
        stats["steps"] = len(trajectories)
    else:
        # GRPO: group relative policy optimization
        for i, traj in enumerate(trajectories[:GRPO_BATCH_SIZE]):
            print(f"[trainer] GRPO step {i+1}/{min(GRPO_BATCH_SIZE, len(trajectories))}")

            # Build prompt
            prompt = (
                f"Question: {traj.question}\n\n"
                f"Context:\n{traj.context[:1500]}\n\nAnswer:"
            )

            # Generate group of responses
            group_responses = generate_group_responses(model, tokenizer, prompt)

            # Score each response
            responses_with_rewards = []
            for response_text, log_probs in group_responses:
                # Create temp trajectory for scoring
                temp_traj = Trajectory(
                    question=traj.question,
                    retrieved_paths=traj.retrieved_paths,
                    context=traj.context,
                    response=response_text,
                    citations=[],
                    citation_validation={},
                )
                reward = compute_full_reward(temp_traj, compute_coverage=False)
                responses_with_rewards.append((response_text, log_probs, reward))

            # Compute and apply GRPO loss
            optimizer.zero_grad()
            loss = grpo_loss(model, tokenizer, traj, responses_with_rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(loss.item())
            stats["steps"] += 1

    stats["mean_loss"] = float(np.mean(losses)) if losses else 0.0

    # Save checkpoint
    step_count = _count_total_steps() + stats["steps"]
    ckpt_path = LORA_CHECKPOINTS_DIR / f"step-{step_count:06d}"
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"[trainer] Checkpoint saved: {ckpt_path}")

    # Mark trajectories as trained
    mark_trajectories_trained([t.id for t in trajectories])

    # Clear the training-suggested flag so the next threshold crossing re-prompts
    from lore.evolve.trajectory import TRAINING_SUGGESTED_FLAG
    TRAINING_SUGGESTED_FLAG.unlink(missing_ok=True)

    # Check for reward divergence
    _check_divergence_guard()

    # Notify serve.py to hot-swap the adapter
    _signal_hot_swap(ckpt_path)

    stats["checkpoint"] = str(ckpt_path)
    print(f"[trainer] Done: {stats}")
    return stats


def _count_total_steps() -> int:
    """Count total training steps from checkpoint names."""
    LORA_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = list(LORA_CHECKPOINTS_DIR.glob("step-*"))
    if not checkpoints:
        return 0
    try:
        return max(int(c.name.split("-")[1]) for c in checkpoints)
    except (IndexError, ValueError):
        return len(checkpoints) * GRPO_BATCH_SIZE


def _check_divergence_guard() -> None:
    """Rollback to previous checkpoint if reward is degrading."""
    stats = get_reward_stats()
    if stats.get("total", 0) < 50:
        return  # Not enough data

    recent_mean = stats.get("recent_mean", 0)
    all_mean = stats.get("all_mean", 0)
    all_std = stats.get("all_std", 0)

    if recent_mean < all_mean - all_std and all_std > 0.05:
        print(f"[trainer] DIVERGENCE GUARD: recent_mean={recent_mean:.3f} < "
              f"all_mean-std={all_mean - all_std:.3f} — rolling back!")
        _rollback_to_previous_checkpoint()


def _rollback_to_previous_checkpoint() -> None:
    """Delete the most recent checkpoint to roll back."""
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    if len(checkpoints) >= 2:
        import shutil
        bad = checkpoints[-1]
        shutil.rmtree(str(bad))
        print(f"[trainer] Rolled back: deleted {bad}")


def _signal_hot_swap(ckpt_path: Path) -> None:
    """Write a signal file for serve.py to pick up."""
    signal_file = LORA_CHECKPOINTS_DIR / ".new_checkpoint"
    signal_file.write_text(str(ckpt_path))


if __name__ == "__main__":
    background = "--background" in sys.argv
    run_training(background=background)
