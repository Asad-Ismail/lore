"""Capture and store question traces for curiosity training."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from sqlitedict import SqliteDict

from lore.config import QUESTION_TRACES_DB, CURIOSITY_TRAIN_THRESHOLD, DATA_DIR


@dataclass
class QuestionTrace:
    """Records a user question alongside the wiki state when it was asked."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    wiki_state: str = ""
    question: str = ""
    reward: float = -1.0
    trained: bool = False
    metadata: dict = field(default_factory=dict)


def save_question_trace(trace: QuestionTrace) -> None:
    QUESTION_TRACES_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDict(str(QUESTION_TRACES_DB), autocommit=True) as db:
        db[trace.id] = asdict(trace)


def capture_question_trace(question: str, wiki_state: str) -> QuestionTrace:
    """Record a user's question + current wiki state for curiosity training."""
    trace = QuestionTrace(question=question, wiki_state=wiki_state)
    save_question_trace(trace)
    stats = get_question_trace_stats()
    untrained = stats.get("untrained", 0)
    remaining = max(0, CURIOSITY_TRAIN_THRESHOLD - untrained)
    if remaining > 0:
        print(f"[curiosity] Trace saved ({trace.id[:8]}) — {remaining} more until training")
    else:
        print(f"[curiosity] Trace saved ({trace.id[:8]}) — ready to train ({untrained} untrained)")
    _maybe_suggest_curiosity_training()
    return trace


def get_all_question_traces(only_untrained: bool = False) -> list[QuestionTrace]:
    if not QUESTION_TRACES_DB.exists():
        return []
    results = []
    with SqliteDict(str(QUESTION_TRACES_DB)) as db:
        for v in db.values():
            if only_untrained and v.get("trained", False):
                continue
            results.append(QuestionTrace(**v))
    results.sort(key=lambda t: t.timestamp)
    return results


def get_all_past_questions() -> list[str]:
    """Return all questions the user has ever asked."""
    if not QUESTION_TRACES_DB.exists():
        return []
    questions = []
    with SqliteDict(str(QUESTION_TRACES_DB)) as db:
        for v in db.values():
            q = v.get("question", "")
            if q:
                questions.append(q)
    return list(set(questions))


def get_question_trace_stats() -> dict:
    if not QUESTION_TRACES_DB.exists():
        return {"total": 0, "untrained": 0}
    total = untrained = 0
    with SqliteDict(str(QUESTION_TRACES_DB)) as db:
        for v in db.values():
            total += 1
            if not v.get("trained", False):
                untrained += 1
    return {"total": total, "untrained": untrained}


def mark_question_traces_trained(trace_ids: list[str]) -> None:
    with SqliteDict(str(QUESTION_TRACES_DB), autocommit=True) as db:
        for tid in trace_ids:
            if tid in db:
                entry = dict(db[tid])
                entry["trained"] = True
                db[tid] = entry


CURIOSITY_SUGGESTED_FLAG = DATA_DIR / ".curiosity_suggested"


def _maybe_suggest_curiosity_training() -> None:
    stats = get_question_trace_stats()
    untrained = stats.get("untrained", 0)
    if untrained < CURIOSITY_TRAIN_THRESHOLD:
        return
    if CURIOSITY_SUGGESTED_FLAG.exists():
        return
    CURIOSITY_SUGGESTED_FLAG.parent.mkdir(parents=True, exist_ok=True)
    CURIOSITY_SUGGESTED_FLAG.write_text(str(untrained))
    print(
        f"\n[lore-train] {untrained} question traces collected.\n"
        f"Run `lore-train curiosity` to train the model on your questioning patterns.\n"
    )
