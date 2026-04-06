"""Generate Marp slide decks from wiki content."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from lore.config import OUTPUTS_DIR, WIKI_DIR
from lore.index.search import hybrid_search
from lore.compile.compiler import generate


def generate_slides(topic: str, max_slides: int = 20) -> Path:
    """Generate a Marp-compatible markdown slide deck on a topic."""
    print(f"[render] Generating slides: {topic}")
    slides_dir = OUTPUTS_DIR / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    # Gather content
    results = hybrid_search(topic, top_k=8)
    if not results:
        print(f"[warn] No wiki articles found for: {topic}")

    context_parts = []
    for r in results[:6]:
        full = (WIKI_DIR / r.article_path).read_text(encoding="utf-8", errors="replace")
        context_parts.append(f"=== {r.title} ===\n{full[:2000]}")

    context = "\n\n".join(context_parts)

    # Ask LLM to produce Marp slides
    prompt = (
        f"Wiki context about {topic}:\n\n{context}\n\n"
        f"---\n\n"
        f"Create a Marp slide deck about '{topic}' with {min(max_slides, 12)} slides.\n"
        f"Format: Each slide separated by '---' on its own line.\n"
        f"First slide: title + subtitle. Remaining slides: one concept per slide.\n"
        f"Use bullet points. Include speaker notes as <!-- comment --> after each slide.\n"
        f"Do NOT include the Marp frontmatter (---\\nmarp: true...) — that will be added.\n"
        f"Output only the slide content."
    )
    slides_content = generate(prompt)

    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^\w]+", "-", topic.lower())[:40]
    filename = f"{slug}-{now.strftime('%Y%m%d')}.md"
    output_path = slides_dir / filename

    marp_header = f"""---
marp: true
theme: default
paginate: true
footer: '{topic} | Personal Research Wiki | {now.strftime("%Y-%m-%d")}'
---

"""

    output_path.write_text(marp_header + slides_content, encoding="utf-8")
    print(f"[render] Slides saved: {output_path}")
    return output_path
