"""RAG-based Q&A agent over the wiki."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lore.config import WIKI_DIR, OUTPUTS_DIR, QUERY_SYSTEM_PROMPT, MAX_NEW_TOKENS
from lore.index.search import search_and_read, SearchResult
from lore.query.citation import extract_citations, validate_citations
from lore.compile.compiler import generate


@dataclass
class QueryResult:
    question: str
    answer: str
    retrieved_paths: list[str]
    context_used: str
    citations: list[str]
    citation_validation: dict
    output_path: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def answer_question(
    question: str,
    top_k: int = 8,
    save_output: bool = True,
    capture_trajectory: bool = True,
) -> QueryResult:
    """
    Full RAG pipeline: search → assemble context → generate answer → save.
    """
    print(f"[query] Searching wiki for: {question!r}")

    # 1. Retrieve relevant articles
    search_results = search_and_read(question, top_k=top_k)

    retrieved_paths = [r.article_path for r, _ in search_results]
    print(f"[query] Retrieved {len(search_results)} articles: {[r.title for r, _ in search_results]}")

    # 2. Assemble context
    context_parts = []
    for result, content in search_results:
        context_parts.append(f"=== {result.title} (wiki/{result.article_path}) ===\n{content[:2000]}")

    # 3. Fallback: if search returned nothing, use _summaries.md as coarse context
    if not context_parts:
        summaries_path = WIKI_DIR / "_summaries.md"
        if summaries_path.exists():
            summary_text = summaries_path.read_text(encoding="utf-8", errors="replace")[:4000]
            context = f"[No direct search results. Wiki article summaries:\n\n{summary_text}]"
        else:
            context = "[No relevant wiki articles found for this question]"
    else:
        context = "\n\n".join(context_parts)

    prompt = (
        f"Wiki context:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = generate(prompt, system=QUERY_SYSTEM_PROMPT)

    # 4. Extract and validate citations
    citations = extract_citations(answer)
    citation_validation = validate_citations(citations, retrieved_paths)

    # 5. Build output report
    output_path = ""
    if save_output:
        output_path = _save_query_report(question, answer, search_results, citation_validation)

    result = QueryResult(
        question=question,
        answer=answer,
        retrieved_paths=retrieved_paths,
        context_used=context,
        citations=citations,
        citation_validation=citation_validation,
        output_path=output_path,
    )

    # 6. Capture trajectory for RL training
    if capture_trajectory:
        try:
            from lore.evolve.trajectory import capture_query_trajectory
            capture_query_trajectory(result)
        except Exception as e:
            print(f"[warn] Trajectory capture failed: {e}")

    return result


def _save_query_report(
    question: str,
    answer: str,
    search_results: list[tuple[SearchResult, str]],
    citation_validation: dict,
) -> str:
    """Write the Q&A result to outputs/queries/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    queries_dir = OUTPUTS_DIR / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^\w]+", "-", question.lower())[:50].strip("-")
    filename = f"{timestamp}-{slug}.md"
    output_path = queries_dir / filename

    # Build citation table
    cite_rows = []
    valid = set(citation_validation.get("valid", []))
    hallucinated = set(citation_validation.get("hallucinated", []))
    nonexistent = set(citation_validation.get("nonexistent", []))
    all_cites = valid | hallucinated | nonexistent
    for cite in sorted(all_cites):
        if cite in valid:
            status = "Valid"
        elif cite in hallucinated:
            status = "Hallucinated (not retrieved)"
        else:
            status = "Does not exist"
        cite_rows.append(f"| [[{cite}]] | {status} |")

    # Build retrieved articles section
    retrieved_section = "\n".join(
        f"- [{r.title}](../wiki/{r.article_path}) (score: {r.score:.4f})"
        for r, _ in search_results
    )

    report = f"""# Query: {question}
Date: {now.isoformat()}

## Answer

{answer}

## Retrieved Articles

{retrieved_section or "_None_"}

## Citation Validation

| Citation | Status |
|---|---|
{chr(10).join(cite_rows) if cite_rows else "| _No citations_ | — |"}

## Related Topics

<!-- Add follow-up questions or related wiki topics here -->
"""

    output_path.write_text(report, encoding="utf-8")
    print(f"[query] Report saved: {output_path.relative_to(OUTPUTS_DIR.parent)}")
    return str(output_path)
