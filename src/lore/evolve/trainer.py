"""
Curiosity training: teaches the local model to ask questions like the user.

Architecture:
- Base model: Qwen/Qwen3-1.7B (~4 GB VRAM in bf16)
- LoRA adapter: r=16, alpha=32, targets q/k/v/o_proj (~50 MB)
- Phase 1 (SFT): imitate user questions given wiki state
- Phase 2 (GRPO): optimize 4-signal question reward
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lore.config import (
    LORA_BASE_MODEL_ID, HF_CACHE_DIR, LORA_CHECKPOINTS_DIR,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    CURIOSITY_GROUP_SIZE, GRPO_BATCH_SIZE,
    LEARNING_RATE, MAX_GRAD_NORM, get_device_map, get_torch_dtype,
)


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
        model = PeftModel.from_pretrained(base, str(latest_ckpt), is_trainable=True)
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
    LORA_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    return checkpoints[-1] if checkpoints else None



def run_curiosity_training() -> dict:
    """
    Train the model to generate questions like the user.
    Phase 1 (< CURIOSITY_BOOTSTRAP_N traces): SFT on user questions.
    Phase 2 (>= CURIOSITY_BOOTSTRAP_N traces): GRPO with question reward.
    """
    from lore.config import CURIOSITY_BOOTSTRAP_N
    from lore.evolve.trajectory import (
        get_all_question_traces, mark_question_traces_trained,
        get_all_past_questions, CURIOSITY_SUGGESTED_FLAG,
    )
    from lore.evolve.curiosity import (
        question_reward, build_wiki_state_summary, CURIOSITY_SYSTEM_PROMPT,
    )

    print(f"[curiosity] Starting curiosity training at {datetime.now(timezone.utc).isoformat()}")

    traces = get_all_question_traces(only_untrained=True)
    if not traces:
        print("[curiosity] No question traces to train on.")
        return {"steps": 0, "reason": "no_traces"}

    all_traces = get_all_question_traces()
    use_sft = len(all_traces) < CURIOSITY_BOOTSTRAP_N
    print(f"[curiosity] Mode: {'SFT (imitation)' if use_sft else 'GRPO (RL)'}")
    print(f"[curiosity] Training on {len(traces)} question traces")

    model, tokenizer = load_student_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    stats = {"steps": 0, "mean_loss": 0.0, "mode": "sft" if use_sft else "grpo"}
    losses = []
    past_questions = get_all_past_questions()

    if use_sft:
        for trace in traces[:20]:
            prompt = (
                f"Wiki state:\n{trace.wiki_state[:1500]}\n\n"
                f"Generate the question this researcher would ask:"
            )
            full_text = prompt + " " + trace.question

            inputs = tokenizer(
                full_text, return_tensors="pt", max_length=1024, truncation=True,
            ).to(model.device)

            labels = inputs["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[0, :prompt_len] = -100

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(loss.item())
            stats["steps"] += 1
    else:
        wiki_state = build_wiki_state_summary()
        for trace in traces[:GRPO_BATCH_SIZE]:
            prompt = (
                f"Wiki state:\n{trace.wiki_state[:1500]}\n\n"
                f"Generate a follow-up question this researcher should explore:"
            )

            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": CURIOSITY_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            candidates_with_rewards = []
            for _ in range(CURIOSITY_GROUP_SIZE):
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=100, do_sample=True,
                        temperature=0.9, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )

                gen_ids = output.sequences[0][input_len:]
                candidate = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                candidate = candidate.split("\n")[0].strip()

                reward_dict = question_reward(candidate, wiki_state, past_questions)
                candidates_with_rewards.append((gen_ids, candidate, reward_dict["combined"]))

            rewards = torch.tensor([r for _, _, r in candidates_with_rewards])
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=model.device)
            count = 0

            for (gen_ids, _, _), advantage in zip(candidates_with_rewards, advantages):
                if gen_ids.numel() == 0:
                    continue
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
                outputs = model(input_ids=full_ids)
                logits = outputs.logits[0, input_len - 1:-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                total_loss = total_loss - advantage.to(model.device) * token_lp.mean()
                count += 1

            loss = total_loss / max(count, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(loss.item())
            stats["steps"] += 1

    stats["mean_loss"] = float(np.mean(losses)) if losses else 0.0

    step_count = _count_total_steps() + stats["steps"]
    ckpt_path = LORA_CHECKPOINTS_DIR / f"step-{step_count:06d}"
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    print(f"[curiosity] Checkpoint saved: {ckpt_path}")

    mark_question_traces_trained([t.id for t in traces])
    CURIOSITY_SUGGESTED_FLAG.unlink(missing_ok=True)
    stats["checkpoint"] = str(ckpt_path)
    print(f"[curiosity] Done: {stats}")
    return stats


def _count_total_steps() -> int:
    LORA_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = list(LORA_CHECKPOINTS_DIR.glob("step-*"))
    if not checkpoints:
        return 0
    try:
        return max(int(c.name.split("-")[1]) for c in checkpoints)
    except (IndexError, ValueError):
        return len(checkpoints) * GRPO_BATCH_SIZE


def rollback_checkpoint(n: int = 1) -> None:
    """Delete the most recent N checkpoints."""
    import shutil
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    for ckpt in checkpoints[-n:]:
        shutil.rmtree(str(ckpt))
        print(f"[trainer] Rolled back: deleted {ckpt}")
