"""Replay buffer for RL training with curriculum sampling."""

from __future__ import annotations

import random
from typing import Literal

from sqlitedict import SqliteDict

from lore.config import TRAJECTORIES_DB, TRAIN_BUFFER_SAMPLE
from lore.evolve.trajectory import Trajectory


SampleStrategy = Literal["recent", "high_reward", "low_reward", "mixed", "curriculum"]


def sample_trajectories(
    n: int = TRAIN_BUFFER_SAMPLE,
    strategy: SampleStrategy = "curriculum",
    min_reward: float = 0.0,
) -> list[Trajectory]:
    """
    Sample trajectories from the buffer using the given strategy.

    Strategies:
    - recent: last N trajectories
    - high_reward: top N by reward
    - low_reward: bottom N by reward (contrastive signal)
    - mixed: 50% recent + 25% high + 25% low
    - curriculum: adapts based on current training stage
    """
    if not TRAJECTORIES_DB.exists():
        return []

    all_trajs: list[Trajectory] = []
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        for v in db.values():
            t = Trajectory(**v)
            if t.reward >= min_reward:
                all_trajs.append(t)

    if not all_trajs:
        return []

    all_trajs.sort(key=lambda t: t.timestamp)

    if strategy == "recent":
        return all_trajs[-n:]

    if strategy == "high_reward":
        return sorted(all_trajs, key=lambda t: t.reward, reverse=True)[:n]

    if strategy == "low_reward":
        # Only include low-reward that aren't degenerate
        valid = [t for t in all_trajs if t.reward > 0.1]
        return sorted(valid, key=lambda t: t.reward)[:n]

    if strategy == "mixed":
        recent = all_trajs[-max(1, n // 2):]
        high = sorted(all_trajs, key=lambda t: t.reward, reverse=True)[:max(1, n // 4)]
        low = sorted([t for t in all_trajs if t.reward > 0.1], key=lambda t: t.reward)[:max(1, n // 4)]
        combined = list({t.id: t for t in recent + high + low}.values())
        return combined[:n]

    if strategy == "curriculum":
        return _curriculum_sample(all_trajs, n)

    # Default: random
    return random.sample(all_trajs, min(n, len(all_trajs)))


def _curriculum_sample(trajs: list[Trajectory], n: int) -> list[Trajectory]:
    """
    Curriculum: adapt difficulty based on training progress.
    - Early stage (< 100 trajs): prefer high-reward (easy wins)
    - Mid stage (100-500): prefer medium-reward (most learning signal)
    - Late stage (> 500): uniform mix
    """
    total = len(trajs)

    if total < 100:
        # Early: focus on high-reward for basic competence
        return sorted(trajs, key=lambda t: t.reward, reverse=True)[:n]
    elif total < 500:
        # Mid: focus on medium difficulty (0.3–0.7 reward range)
        medium = [t for t in trajs if 0.3 <= t.reward <= 0.7]
        if len(medium) >= n:
            return random.sample(medium, n)
        # Fill with high-reward
        high = [t for t in trajs if t.reward > 0.7]
        combined = medium + high
        return combined[:n] if len(combined) >= n else combined
    else:
        # Late: uniform mix
        return random.sample(trajs, min(n, len(trajs)))


def mark_trajectories_trained(trajectory_ids: list[str]) -> None:
    """Mark trajectories as used for training."""
    with SqliteDict(str(TRAJECTORIES_DB), autocommit=True) as db:
        for tid in trajectory_ids:
            if tid in db:
                entry = dict(db[tid])
                entry["trained"] = True
                db[tid] = entry


def get_reward_stats() -> dict:
    """Return reward distribution stats for monitoring."""
    import numpy as np

    if not TRAJECTORIES_DB.exists():
        return {}

    rewards = []
    with SqliteDict(str(TRAJECTORIES_DB)) as db:
        for v in db.values():
            r = v.get("reward", -1.0)
            if r >= 0:
                rewards.append(r)

    if not rewards:
        return {}

    arr = np.array(rewards)
    # Rolling stats (last 100)
    recent = arr[-100:] if len(arr) > 100 else arr

    return {
        "total": len(arr),
        "all_mean": float(arr.mean()),
        "all_std": float(arr.std()),
        "recent_mean": float(recent.mean()),
        "recent_std": float(recent.std()),
        "recent_min": float(recent.min()),
        "recent_max": float(recent.max()),
    }
