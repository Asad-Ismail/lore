"""Capture and store Q&A trajectories for RL training."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from sqlitedict import SqliteDict

from lore.config import TRAJECTORIES_DB, TRAIN_THRESHOLD, DATA_DIR


@dataclass
class Trajectory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    question: str = ""
    retrieved_paths: list = field(default_factory=list)
    context: str = ""
    response: str = ""
    citations: list = field(default_factory=list)
    citation_validation: dict = field(default_factory=dict)
    reward: float = -1.0          # -1 = not yet scored
    coverage_score: float = -1.0  # -1 = not yet judged (async)
    trained: bool = False
    metadata: dict = field(default_factory=dict)


def save_trajectory(traj: Trajectory) -> None:
    """Persist a trajectory to the SQLite DB."""
    TRAJECTORIES_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDict(str(TRAJECTORIES_DB), autocommit=True) as db:
        db[traj.id] = asdict(traj)


def capture_query_trajectory(query_result) -> Trajectory:
    """
    Create a Trajectory from a QueryResult and compute initial rewards.
    Saves to DB and triggers training if threshold reached.
    """
    from lore.evolve.reward import compute_instant_rewards

    traj = Trajectory(
        question=query_result.question,
        retrieved_paths=query_result.retrieved_paths,
        context=query_result.context_used,
        response=query_result.answer,
        citations=query_result.citations,
        citation_validation=query_result.citation_validation,
    )

    # Compute instant rewards (grounding + citation + fluency — no LLM needed)
    instant = compute_instant_rewards(traj)
    traj.reward = instant["combined_partial"]  # Partial reward without coverage
    traj.metadata["grounding"] = instant["grounding"]
    traj.metadata["citation"] = instant["citation"]
    traj.metadata["fluency"] = instant["fluency"]

    save_trajectory(traj)
    print(f"[trajectory] Saved {traj.id[:8]} | reward={traj.reward:.3f}")

    # Check if enough new data has accumulated — suggest retraining, don't auto-trigger
    _maybe_suggest_training()

    return traj


# Flag file written when training is suggested; consumed when user runs lore-train train
TRAINING_SUGGESTED_FLAG = DATA_DIR / ".training_suggested"


def _maybe_suggest_training() -> None:
    """
    When untrained trajectories cross TRAIN_THRESHOLD, write a flag file and
    print a suggestion to stdout. Claude Code hooks surface stdout to the
    conversation, so the user sees the prompt and can decide whether to retrain.
    Training is NEVER spawned automatically — the user stays in control.
    """
    untrained = count_untrained_trajectories()
    if untrained < TRAIN_THRESHOLD:
        return

    # Only suggest once per threshold crossing (not on every subsequent query)
    if TRAINING_SUGGESTED_FLAG.exists():
        return

    TRAINING_SUGGESTED_FLAG.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_SUGGESTED_FLAG.write_text(str(untrained))

    # This prints to stdout → captured by the PostToolUse hook → surfaced in conversation
    print(
        f"\n[lore-train] {untrained} new trajectories collected since last training run.\n"
        f"Run `lore-train train` to retrain the LoRA model on this data, or skip to keep collecting.\n"
        f"Check current reward stats with `lore-train status`."
    )


def count_untrained_trajectories() -> int:
    """Count trajectories that haven't been used for training yet."""
    if not TRAJECTORIES_DB.exists():
        return 0
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        return sum(1 for v in db.values() if not v.get("trained", False))


def get_all_trajectories(only_trained: bool = False) -> list[Trajectory]:
    if not TRAJECTORIES_DB.exists():
        return []
    results = []
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        for v in db.values():
            if only_trained and not v.get("trained", False):
                continue
            results.append(Trajectory(**v))
    return results


def get_trajectory_stats() -> dict:
    if not TRAJECTORIES_DB.exists():
        return {"total": 0, "untrained": 0, "mean_reward": 0.0, "std_reward": 0.0}

    rewards = []
    total = untrained = 0
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        for v in db.values():
            total += 1
            if not v.get("trained", False):
                untrained += 1
            r = v.get("reward", -1.0)
            if r >= 0:
                rewards.append(r)

    import numpy as np
    return {
        "total": total,
        "untrained": untrained,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }


def load_latest_trajectory() -> Trajectory | None:
    """Return the most recently saved trajectory, or None."""
    if not TRAJECTORIES_DB.exists():
        return None
    latest = None
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        for v in db.values():
            t = Trajectory(**v)
            if latest is None or t.timestamp > latest.timestamp:
                latest = t
    return latest
