from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parent / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))

WORKSPACE_ROOT = Path(os.environ.get("LORE_SPACE_WORKSPACE", "/tmp/lore-space-workspace")).resolve()
os.environ.setdefault("LORE_REPO_ROOT", str(WORKSPACE_ROOT))

import gradio as gr

from lore.config import WIKI_DIR
from lore.demo import ensure_demo_workspace, ingest_demo_source, seed_demo, workspace_snapshot
from lore.index.store import load_all_articles

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --paper: #f6f2e8;
  --ink: #13231a;
  --accent: #0d6b6f;
  --accent-soft: #d8efe7;
  --sand: #ead8bc;
}

body,
.gradio-container {
  font-family: 'IBM Plex Sans', sans-serif !important;
  background:
    radial-gradient(circle at top left, rgba(234, 216, 188, 0.65), transparent 32rem),
    radial-gradient(circle at bottom right, rgba(13, 107, 111, 0.16), transparent 28rem),
    var(--paper);
  color: var(--ink);
}

.gradio-container {
  max-width: 1200px !important;
}

.hero {
  padding: 1.25rem 1.4rem 1.4rem 1.4rem;
  border: 1px solid rgba(19, 35, 26, 0.08);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.62);
  backdrop-filter: blur(10px);
  box-shadow: 0 20px 50px rgba(19, 35, 26, 0.07);
}

.hero h1 {
  margin: 0;
  font-size: 2.3rem;
  line-height: 1.05;
}

.hero p {
  margin: 0.85rem 0 0 0;
  max-width: 55rem;
  font-size: 1.02rem;
}

.eyebrow {
  display: inline-block;
  margin-bottom: 0.9rem;
  padding: 0.28rem 0.55rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.panel {
  border: 1px solid rgba(19, 35, 26, 0.08);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.68);
  box-shadow: 0 16px 40px rgba(19, 35, 26, 0.05);
}

.mono code,
.mono pre,
.mono {
  font-family: 'IBM Plex Mono', monospace !important;
}
"""

SAMPLE_URLS = [
    ["https://arxiv.org/abs/2306.00978"],
    ["https://modelcontextprotocol.io/introduction"],
]


def _relative_to_workspace(path_str: str) -> str:
    path = Path(path_str)
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _status_markdown() -> str:
    ensure_demo_workspace()
    snapshot = workspace_snapshot()
    latest = snapshot["latest_article"] or "_none yet_"
    return (
        "### Workspace\n\n"
        f"- Root: `{WORKSPACE_ROOT}`\n"
        f"- Articles: **{snapshot['article_count']}**\n"
        f"- Sources: **{snapshot['source_count']}**\n"
        f"- Latest page: `{latest}`"
    )


def _library_markdown() -> str:
    ensure_demo_workspace()
    articles = load_all_articles(WIKI_DIR)
    if not articles:
        return "### Wiki Pages\n\n_No articles yet._"

    recent = articles[-8:]
    lines = ["### Wiki Pages", ""]
    for article in reversed(recent):
        lines.append(f"- `{article.category}/{article.title}.md`")
    return "\n".join(lines)


def _suggestions_markdown(suggestions: list[dict], mode: str) -> str:
    if not suggestions:
        return "### Next Questions\n\n_No follow-up questions available yet._"

    lines = [
        "### Next Questions",
        "",
        f"_Mode: `{mode}`_",
        "",
    ]
    for suggestion in suggestions:
        lines.append(f"- {suggestion['question']}")
    return "\n".join(lines)


def _article_markdown(result) -> str:
    raw_path = _relative_to_workspace(result.raw_path)
    return (
        "### New Page\n\n"
        f"- Action: **{result.action}**\n"
        f"- Source: `{raw_path}`\n"
        f"- Page: `{result.article_path}`\n\n"
        f"{result.article_content}"
    )


def ingest_workflow(url: str, upload_path: str | None):
    url = (url or "").strip()
    if not url and not upload_path:
        raise gr.Error("Provide one source: either a URL or a PDF/Markdown upload.")
    if url and upload_path:
        raise gr.Error("Use one source at a time so the demo stays legible.")

    try:
        if url:
            result = ingest_demo_source(url, kind="url")
        else:
            result = ingest_demo_source(upload_path, kind="file")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    return (
        _status_markdown(),
        _article_markdown(result),
        _suggestions_markdown(result.suggestions, result.suggestion_mode),
        _library_markdown(),
    )


def reset_workspace():
    seed_demo(reset=True)
    return (
        _status_markdown(),
        "### New Page\n\n_Reset the demo workspace. Ingest a URL or upload a source to create a fresh page._",
        "### Next Questions\n\n_The starter corpus is back in place. Run one ingest to generate the next set of suggestions._",
        _library_markdown(),
    )


ensure_demo_workspace()

with gr.Blocks(title="Lore Demo") as demo:
    gr.Markdown(
        """
        <div class="hero">
          <div class="eyebrow">Lore x Hugging Face Space</div>
          <h1>Ingest one source. Write one wiki page. Get three next questions.</h1>
          <p>
            This demo runs Lore against a seeded writable workspace. It ingests a URL or uploaded PDF,
            writes a single markdown page into the wiki, rebuilds the index, and returns follow-up
            questions using the same deterministic curiosity stack that powers fresh local clones.
          </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                source_url = gr.Textbox(
                    label="URL",
                    placeholder="https://arxiv.org/abs/... or any article URL",
                )
                upload = gr.File(
                    label="PDF or Markdown upload",
                    file_types=[".pdf", ".md", ".txt"],
                    type="filepath",
                )
                with gr.Row():
                    ingest_button = gr.Button("Ingest Into Lore", variant="primary")
                    reset_button = gr.Button("Reset Demo Workspace")
                gr.Examples(SAMPLE_URLS, inputs=source_url, label="Sample URLs")

        with gr.Column(scale=4):
            status = gr.Markdown(_status_markdown(), elem_classes=["mono"])
            library = gr.Markdown(_library_markdown(), elem_classes=["mono"])

    with gr.Row():
        article = gr.Markdown(
            "### New Page\n\n_Ingest a source to write the next wiki page._",
            elem_classes=["mono"],
        )
        suggestions = gr.Markdown(
            "### Next Questions\n\n_The demo will suggest what to explore next after each ingest._",
            elem_classes=["mono"],
        )

    ingest_button.click(
        ingest_workflow,
        inputs=[source_url, upload],
        outputs=[status, article, suggestions, library],
    )
    reset_button.click(
        reset_workspace,
        outputs=[status, article, suggestions, library],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
