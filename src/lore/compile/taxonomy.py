"""Route wiki articles to the correct category subdirectory."""

from __future__ import annotations

from lore.config import WIKI_CATEGORIES

# Keyword-based fast routing (no LLM call needed for most articles)
_KEYWORD_MAP: dict[str, list[str]] = {
    "techniques": [
        "quantization", "pruning", "distillation", "compression", "gptq", "awq", "gguf",
        "smoothquant", "sparsegpt", "lora", "qlora", "reft", "peft", "fine-tuning",
        "finetuning", "training", "optimization", "gradient", "backprop", "dropout",
        "normalization", "regularization", "calibration", "post-training", "qat", "ptq",
        "weight sharing", "mixed precision", "int4", "int8", "fp16", "bf16",
    ],
    "models": [
        "llama", "qwen", "mistral", "gemma", "phi", "gpt", "bert", "t5", "palm",
        "falcon", "mamba", "rwkv", "olmo", "bloom", "deepseek", "claude", "gemini",
        "architecture", "transformer", "attention", "moe", "mixture of experts",
        "ssm", "state space", "diffusion", "vit", "vision transformer",
    ],
    "papers": [
        "arxiv", "paper", "publication", "preprint", "journal", "conference",
        "neurips", "icml", "iclr", "acl", "emnlp", "cvpr", "iccv", "eccv",
    ],
    "datasets": [
        "dataset", "corpus", "benchmark data", "training data", "evaluation data",
        "wikitext", "pile", "redpajama", "dolma", "openwebtext", "c4",
    ],
    "benchmarks": [
        "benchmark", "evaluation", "leaderboard", "mmlu", "hellaswag", "arc",
        "winogrande", "gsm8k", "humaneval", "mbpp", "lm-eval", "perplexity",
        "throughput", "latency", "memory footprint", "flops",
    ],
    "people": [
        "researcher", "professor", "scientist", "engineer", "phd", "postdoc",
        "university", "lab", "group", "institute",
    ],
    "concepts": [
        "concept", "idea", "theory", "principle", "definition", "overview",
        "introduction", "survey", "review", "tutorial",
        "neural network", "deep learning", "machine learning", "inference",
        "representation", "embedding", "token", "context window", "kv cache",
        "weight", "activation", "loss", "objective", "reward", "policy",
    ],
    "meta": [
        "reading list", "agenda", "roadmap", "plan", "todo", "notes",
        "open problems", "research direction", "future work",
    ],
}


def classify_article(title: str, content_snippet: str = "") -> str:
    """
    Classify an article into a wiki category using keyword matching.
    Falls back to 'concepts' if ambiguous.
    """
    text = (title + " " + content_snippet).lower()

    scores: dict[str, int] = {cat: 0 for cat in WIKI_CATEGORIES}
    for category, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text:
                scores[category] += 1

    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "concepts"


def classify_article_llm(title: str, content_snippet: str, model_fn) -> str:
    """
    Use the LLM to classify an ambiguous article.
    model_fn: callable(prompt: str) -> str
    """
    categories_str = ", ".join(WIKI_CATEGORIES)
    prompt = (
        f"Classify this ML research wiki article into exactly one category.\n"
        f"Categories: {categories_str}\n\n"
        f"Title: {title}\n"
        f"Excerpt: {content_snippet[:400]}\n\n"
        f"Respond with only the category name, nothing else."
    )
    result = model_fn(prompt).strip().lower()
    # Validate response
    for cat in WIKI_CATEGORIES:
        if cat in result:
            return cat
    return "concepts"


def article_path(category: str, title: str) -> str:
    """Convert category + title to a relative wiki path."""
    import re
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return f"{category}/{slug}.md"
