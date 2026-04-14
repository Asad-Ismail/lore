"""
Curiosity module: learns the user's questioning style from their traces.

Trains the local model to generate questions the user would ask,
then surfaces 2-3 suggestions after each interaction.

Reward signals for question quality:
  gap_targeting  (0.35) — targets knowledge gaps (stubs, orphans, under-linked)
  style_match    (0.25) — matches the user's questioning patterns
  novelty        (0.25) — not a repeat of past questions
  specificity    (0.15) — specific, not vague given current wiki depth
"""

from __future__ import annotations

import re
from pathlib import Path

from lore.config import (
    WIKI_DIR,
    LORA_CHECKPOINTS_DIR,
    CURIOSITY_REWARD_WEIGHT_GAP,
    CURIOSITY_REWARD_WEIGHT_STYLE,
    CURIOSITY_REWARD_WEIGHT_NOVELTY,
    CURIOSITY_REWARD_WEIGHT_SPECIFICITY,
)


# ── Wiki state summary ───────────────────────────────────────────────────────

def build_wiki_state_summary(max_chars: int = 3000) -> str:
    """
    Build a compact representation of the wiki's current state.
    Reads _index.md + stubs + last 5 log entries. Used as the input
    prompt for question generation: "given this wiki, what would the user ask?"
    """
    parts = []

    index_path = WIKI_DIR / "_index.md"
    if index_path.exists():
        index_text = index_path.read_text(encoding="utf-8", errors="replace")
        parts.append(f"=== Wiki Index ===\n{index_text[:max_chars // 2]}")

    stubs = _find_stub_concepts()
    if stubs:
        parts.append(f"=== Knowledge Gaps (stubs) ===\n" + "\n".join(f"- {s}" for s in stubs[:20]))

    log_path = WIKI_DIR / "_log.md"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        entries = re.findall(r"^## \[.*$", log_text, re.MULTILINE)
        if entries:
            parts.append(f"=== Recent Activity ===\n" + "\n".join(entries[-5:]))

    return "\n\n".join(parts) if parts else "[Empty wiki]"


def _find_stub_concepts() -> list[str]:
    """Find [[WikiLink]] targets with no corresponding article."""
    existing = set()
    all_links: set[str] = set()

    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        existing.add(_normalize(md_file.stem))
        content = md_file.read_text(encoding="utf-8", errors="replace")
        for link in re.findall(r"\[\[([^\]|#]+?)\]\]", content):
            all_links.add(link)

    return [link for link in sorted(all_links) if _normalize(link) not in existing]


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


# ── Question reward function ─────────────────────────────────────────────────

def question_reward(
    question: str,
    wiki_state: str,
    past_questions: list[str],
) -> dict:
    """
    Score a candidate question with 4 signals.
    Returns dict with individual scores and combined reward.
    """
    gap = _gap_targeting_reward(question, wiki_state)
    style = _style_similarity_reward(question, past_questions)
    novelty = _novelty_reward(question, past_questions)
    specificity = _specificity_reward(question, wiki_state)

    combined = (
        CURIOSITY_REWARD_WEIGHT_GAP * gap
        + CURIOSITY_REWARD_WEIGHT_STYLE * style
        + CURIOSITY_REWARD_WEIGHT_NOVELTY * novelty
        + CURIOSITY_REWARD_WEIGHT_SPECIFICITY * specificity
    )

    return {
        "gap_targeting": gap,
        "style_match": style,
        "novelty": novelty,
        "specificity": specificity,
        "combined": combined,
    }


def _gap_targeting_reward(question: str, wiki_state: str) -> float:
    """
    Higher reward if the question references concepts that are stubs
    or appear in the "Knowledge Gaps" section of the wiki state.
    """
    if "Knowledge Gaps" not in wiki_state:
        return 0.5

    gaps_section = wiki_state.split("Knowledge Gaps")[1] if "Knowledge Gaps" in wiki_state else ""
    gap_concepts = re.findall(r"- (.+)", gaps_section)
    if not gap_concepts:
        return 0.5

    q_lower = question.lower()
    hits = sum(1 for concept in gap_concepts if concept.lower().strip() in q_lower)
    return min(1.0, hits / max(1, min(3, len(gap_concepts))))


def _style_similarity_reward(question: str, past_questions: list[str]) -> float:
    """
    TF-IDF cosine similarity between the candidate question and the
    user's past questions. Captures vocabulary, length patterns, depth.
    """
    if not past_questions or len(past_questions) < 3:
        return 0.5

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = past_questions + [question]
        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        matrix = vect.fit_transform(corpus)

        candidate_vec = matrix[-1]
        past_matrix = matrix[:-1]
        sims = cosine_similarity(candidate_vec, past_matrix).flatten()

        return float(sims.mean())
    except Exception:
        return 0.5


def _novelty_reward(question: str, past_questions: list[str]) -> float:
    """
    Inverse of max similarity to any past question.
    High reward = the question hasn't been asked before.
    """
    if not past_questions:
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = past_questions + [question]
        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        matrix = vect.fit_transform(corpus)

        candidate_vec = matrix[-1]
        past_matrix = matrix[:-1]
        sims = cosine_similarity(candidate_vec, past_matrix).flatten()

        max_sim = float(sims.max())
        return max(0.0, 1.0 - max_sim)
    except Exception:
        return 0.5


def _specificity_reward(question: str, wiki_state: str) -> float:
    """
    Penalizes vague questions when the wiki has deep coverage.
    Rewards questions that mention specific concepts.
    """
    words = question.split()
    if len(words) < 5:
        return 0.1

    index_section = wiki_state.split("Knowledge Gaps")[0] if "Knowledge Gaps" in wiki_state else wiki_state
    known_concepts = re.findall(r"\[\[([^\]]+)\]\]", index_section)
    known_lower = {c.lower() for c in known_concepts}

    q_lower = question.lower()
    concept_mentions = sum(1 for c in known_lower if c in q_lower)

    length_score = min(1.0, len(words) / 15.0)
    concept_score = min(1.0, concept_mentions / 2.0)
    return 0.5 * length_score + 0.5 * concept_score


DAEMON_URL = "http://127.0.0.1:8765"


def _try_daemon_suggest(n: int) -> list[dict] | None:
    """Try to get suggestions from the daemon. Returns None if daemon isn't running."""
    try:
        import httpx
        resp = httpx.get(f"{DAEMON_URL}/suggest", params={"n": n}, timeout=30)
        resp.raise_for_status()
        return resp.json()["suggestions"]
    except Exception:
        return None


def is_daemon_running() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{DAEMON_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ── Suggestion generation ────────────────────────────────────────────────────

CURIOSITY_SYSTEM_PROMPT = """\
You are a research curiosity model. Given the current state of a personal \
knowledge wiki, generate a follow-up question that the user would find \
valuable to explore next. The question should target knowledge gaps, \
build on existing articles, or make connections between concepts. \
Output ONLY the question, nothing else."""


def generate_suggestions(n: int = 3) -> list[dict]:
    suggestions, _ = generate_suggestions_with_mode(n=n)
    return suggestions


def generate_suggestions_with_mode(n: int = 3, prefer_daemon: bool = True) -> tuple[list[dict], str]:
    """
    Generate n candidate questions. Tries the daemon first (~100ms),
    falls back to loading the model directly (~17s).
    """
    if prefer_daemon:
        result = _try_daemon_suggest(n)
        if result is not None:
            return result, "daemon"

    from lore.evolve.trajectory import get_all_past_questions

    wiki_state = build_wiki_state_summary()
    past_questions = get_all_past_questions()

    if wiki_state == "[Empty wiki]":
        return [], "empty"

    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*")) if LORA_CHECKPOINTS_DIR.exists() else []
    if not checkpoints:
        return _heuristic_suggestions(n, wiki_state, past_questions), "heuristic"

    prompt = (
        f"Wiki state:\n{wiki_state}\n\n"
        f"Recent questions asked:\n"
        + "\n".join(f"- {q}" for q in past_questions[-10:])
        + "\n\nGenerate a follow-up question this researcher should explore:"
    )

    try:
        from lore.evolve.trainer import load_student_model
        model, tokenizer = load_student_model()
        model.eval()
    except Exception:
        return _heuristic_suggestions(n, wiki_state, past_questions), "heuristic"

    import torch

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

    seen = set()
    unique = []
    for s in suggestions:
        q_norm = _normalize(s["question"])
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(s)
        if len(unique) >= n:
            break

    if not unique:
        return _heuristic_suggestions(n, wiki_state, past_questions), "heuristic"

    return unique, "checkpoint"


def _heuristic_suggestions(n: int, wiki_state: str, past_questions: list[str]) -> list[dict]:
    """Generate follow-up questions without loading a model checkpoint."""
    from lore.health.suggestions import (
        suggest_connections,
        suggest_new_articles,
        suggest_research_questions,
    )
    from lore.index.store import load_all_articles

    candidates: list[str] = []

    for connection in suggest_connections(max_suggestions=max(8, n * 4)):
        title_a = connection["title_a"]
        title_b = connection["title_b"]
        candidates.append(f"What practical relationship between {title_a} and {title_b} is still missing?")
        candidates.append(f"What is the tradeoff between {title_a} and {title_b} in this workflow?")

    for question in suggest_research_questions():
        candidates.append(question)

    for concept in suggest_new_articles():
        candidates.append(
            f"What should a new article on {concept} cover before I add it to the wiki?"
        )

    article_titles = [article.title for article in load_all_articles()]
    for title in article_titles[:4]:
        candidates.append(f"What is still unresolved in my notes about {title}?")

    if len(article_titles) >= 2:
        candidates.append(
            f"Where does {article_titles[0]} overlap with {article_titles[1]}, and where do they differ?"
        )

    ranked = []
    seen = set()
    for candidate in candidates:
        norm = _normalize(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        reward = question_reward(candidate, wiki_state, past_questions)
        ranked.append({"question": candidate, **reward})

    ranked.sort(key=lambda suggestion: suggestion["combined"], reverse=True)
    return ranked[:n]
