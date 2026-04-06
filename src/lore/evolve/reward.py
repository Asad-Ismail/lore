"""
Reward function for the evolving agent.

Four signals:
  grounding  (0.40) — response grounded in retrieved wiki content
  citation   (0.25) — citations are valid and not hallucinated
  fluency    (0.15) — response is non-degenerate
  coverage   (0.20) — LLM judge scores completeness (async, offline)

Combined partial reward (grounding + citation + fluency) is computed instantly.
Full reward includes coverage and is computed at training time.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lore.config import (
    REWARD_WEIGHT_GROUNDING,
    REWARD_WEIGHT_CITATION,
    REWARD_WEIGHT_COVERAGE,
    REWARD_WEIGHT_FLUENCY,
    GROUNDING_SIM_THRESHOLD,
)

if TYPE_CHECKING:
    from lore.evolve.trajectory import Trajectory


# ── Grounding reward ──────────────────────────────────────────────────────────

def grounding_reward(response: str, context: str) -> float:
    """
    Fraction of response sentences that are grounded in the context.
    Grounded = TF-IDF cosine similarity to at least one context chunk > threshold.
    """
    if not response.strip() or not context.strip():
        return 0.0

    sentences = _split_sentences(response)
    if not sentences:
        return 0.0

    # Split context into chunks (by === separator or paragraphs)
    context_chunks = [c.strip() for c in re.split(r"={3,}.*?={3,}", context) if c.strip()]
    if not context_chunks:
        context_chunks = [context]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_texts = sentences + context_chunks
        vect = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vect.fit_transform(all_texts)

        n_sentences = len(sentences)
        sent_matrix = matrix[:n_sentences]
        chunk_matrix = matrix[n_sentences:]

        sims = cosine_similarity(sent_matrix, chunk_matrix)  # (n_sentences, n_chunks)
        grounded = (sims.max(axis=1) > GROUNDING_SIM_THRESHOLD).sum()
        return float(grounded) / n_sentences

    except Exception:
        # Fallback: keyword overlap
        context_words = set(context.lower().split())
        grounded = sum(
            1 for s in sentences
            if len(set(s.lower().split()) & context_words) / max(1, len(s.split())) > 0.2
        )
        return float(grounded) / len(sentences)


# ── Citation reward ───────────────────────────────────────────────────────────

def citation_reward(citation_validation: dict) -> float:
    """
    Precision of citations: valid / (valid + hallucinated + nonexistent).
    Penalizes citing articles not in context (hallucinated) and nonexistent articles.
    """
    valid = len(citation_validation.get("valid", []))
    hallucinated = len(citation_validation.get("hallucinated", []))
    nonexistent = len(citation_validation.get("nonexistent", []))
    total = valid + hallucinated + nonexistent

    if total == 0:
        # No citations at all — neutral (not penalized but not rewarded)
        return 0.5

    return valid / total


# ── Fluency reward ────────────────────────────────────────────────────────────

def fluency_reward(response: str) -> float:
    """
    Measures response quality:
    - Penalizes too-short responses
    - Penalizes repetition (low unique bigram ratio)
    - Partial penalty for pure hedge responses ("I don't know")
    """
    words = response.lower().split()
    if len(words) < 20:
        return 0.1

    # Unique bigram ratio
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.1
    repetition_score = len(set(bigrams)) / len(bigrams)

    # Hedge penalty: 0.5x if response admits ignorance without content
    hedge_phrases = [
        "i don't know", "i cannot find", "no information",
        "not in the wiki", "not mentioned", "wiki doesn't cover",
    ]
    is_pure_hedge = any(p in response.lower() for p in hedge_phrases) and len(words) < 50
    hedge_factor = 0.6 if is_pure_hedge else 1.0

    return repetition_score * hedge_factor


# ── Coverage reward (LLM judge, offline) ─────────────────────────────────────

def coverage_reward_judge(question: str, response: str, model_fn=None) -> float:
    """
    Use LLM as judge to score response completeness 0–4.
    model_fn: callable(prompt) -> str. If None, loads inference model.
    Returns normalized score 0.0–1.0.
    Cached in trajectories DB by question+response hash.
    """
    import hashlib
    cache_key = hashlib.sha256(f"{question}|||{response}".encode()).hexdigest()[:16]

    # Check cache
    from lore.config import DATA_DIR
    from sqlitedict import SqliteDict
    coverage_cache = DATA_DIR / "coverage_cache.db"
    with SqliteDict(str(coverage_cache)) as db:
        if cache_key in db:
            return db[cache_key]

    if model_fn is None:
        from lore.compile.compiler import generate
        model_fn = generate

    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {response}\n\n"
        f"Score how completely this answer addresses the question on a scale 0–4:\n"
        f"0 = No relevant content, 1 = Partial (< 25%), 2 = Moderate (25-75%), "
        f"3 = Good (75-95%), 4 = Complete (>95%)\n"
        f"Output ONLY the integer (0, 1, 2, 3, or 4)."
    )
    result = model_fn(prompt, system="You are an answer quality evaluator. Output only a single integer.").strip()

    try:
        score = int(re.search(r"\d", result).group())
        score = max(0, min(4, score))
    except (AttributeError, ValueError):
        score = 2  # Default to moderate on parse failure

    normalized = score / 4.0

    with SqliteDict(str(coverage_cache), autocommit=True) as db:
        db[cache_key] = normalized

    return normalized


# ── Combined reward ───────────────────────────────────────────────────────────

def compute_instant_rewards(traj: "Trajectory") -> dict:
    """
    Compute all rewards that don't require an LLM call.
    Returns partial combined reward (without coverage).
    """
    g = grounding_reward(traj.response, traj.context)
    c = citation_reward(traj.citation_validation)
    f = fluency_reward(traj.response)

    # Partial combined (normalize weights without coverage term)
    w_total = REWARD_WEIGHT_GROUNDING + REWARD_WEIGHT_CITATION + REWARD_WEIGHT_FLUENCY
    partial = (
        REWARD_WEIGHT_GROUNDING * g
        + REWARD_WEIGHT_CITATION * c
        + REWARD_WEIGHT_FLUENCY * f
    ) / w_total

    return {
        "grounding": g,
        "citation": c,
        "fluency": f,
        "combined_partial": partial,
    }


def compute_full_reward(traj: "Trajectory", compute_coverage: bool = True) -> float:
    """
    Compute the full 4-signal reward for a trajectory.
    If compute_coverage=True, makes an LLM call for the coverage judge.
    """
    g = traj.metadata.get("grounding", grounding_reward(traj.response, traj.context))
    c = traj.metadata.get("citation", citation_reward(traj.citation_validation))
    f = traj.metadata.get("fluency", fluency_reward(traj.response))

    if compute_coverage:
        cov = coverage_reward_judge(traj.question, traj.response)
    else:
        cov = traj.coverage_score if traj.coverage_score >= 0 else 0.5

    reward = (
        REWARD_WEIGHT_GROUNDING * g
        + REWARD_WEIGHT_CITATION * c
        + REWARD_WEIGHT_COVERAGE * cov
        + REWARD_WEIGHT_FLUENCY * f
    )
    return float(reward)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
