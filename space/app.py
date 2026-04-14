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
from lore.preview import (
    build_preview_articles,
    preview_article_choices,
    render_article_markdown,
    render_graph_html,
    render_library_markdown,
    resolve_preview_path,
)

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --paper: #f6f2e8;
  --ink: #13231a;
  --accent: #0d6b6f;
  --accent-soft: #d8efe7;
  --sand: #ead8bc;
  --rust: #c86a2d;
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
  max-width: 1240px !important;
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
  font-size: 2.35rem;
  line-height: 1.05;
}

.hero p {
  margin: 0.85rem 0 0 0;
  max-width: 58rem;
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

.callout-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 1rem;
}

.callout {
  border: 1px solid rgba(19, 35, 26, 0.08);
  border-radius: 18px;
  padding: 1rem 1.05rem;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 12px 30px rgba(19, 35, 26, 0.05);
}

.callout h3 {
  margin: 0 0 0.45rem 0;
  font-size: 1rem;
}

.callout p,
.callout ul {
  margin: 0;
  font-size: 0.96rem;
  line-height: 1.55;
}

.callout ul {
  padding-left: 1.1rem;
}

.graph-shell {
  display: grid;
  gap: 0.85rem;
}

.graph-summary,
.graph-detail {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  align-items: center;
}

.graph-summary strong {
  margin-right: 0.4rem;
}

.graph-summary span,
.graph-pill {
  padding: 0.35rem 0.6rem;
  border-radius: 999px;
  background: rgba(216, 239, 231, 0.85);
  color: var(--ink);
  font-size: 0.9rem;
}

.graph-pill {
  background: rgba(234, 216, 188, 0.65);
}

.graph-scroll {
  overflow-x: auto;
  border: 1px solid rgba(19, 35, 26, 0.08);
  border-radius: 18px;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.72);
}

.graph-scroll svg {
  width: 100%;
  min-width: 860px;
  height: auto;
  display: block;
}

.graph-column {
  fill: var(--ink);
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.graph-node {
  font-size: 13px;
  font-weight: 600;
}

@media (max-width: 900px) {
  .callout-grid {
    grid-template-columns: 1fr;
  }
}
"""

SAMPLE_URLS = [
    ["https://arxiv.org/abs/2306.00978"],
    ["https://modelcontextprotocol.io/introduction"],
]

PREVIEW_NOTE_HTML = """
<div class="callout-grid">
  <div class="callout">
    <h3>What this Space proves</h3>
    <ul>
      <li>You can ingest one source into a seeded Lore vault with zero setup.</li>
      <li>You can inspect the generated markdown pages, backlinks, and graph shape directly.</li>
      <li>You can see the follow-up question loop before any local checkpoint exists.</li>
    </ul>
  </div>
  <div class="callout">
    <h3>What full Lore adds locally</h3>
    <ul>
      <li>Claude Code or MCP clients write and maintain the wiki over time.</li>
      <li>Local curiosity training can switch from heuristics to Qwen + LoRA checkpoints.</li>
      <li>The full Obsidian vault and agent workflow stay editable on your machine.</li>
    </ul>
  </div>
</div>
"""


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
        "- Mode: **deterministic preview**\n"
        f"- Articles: **{snapshot['article_count']}**\n"
        f"- Sources: **{snapshot['source_count']}**\n"
        f"- Latest page: `{latest}`"
    )


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
        f"- Page: `{result.article_path}`\n"
        "- Writer: **deterministic preview pipeline**\n\n"
        f"{result.article_content}"
    )


def _vault_state(preferred: str | None = None) -> tuple[list[tuple[str, str]], str | None, str, str, str]:
    ensure_demo_workspace()
    preview_articles = build_preview_articles(WIKI_DIR)
    latest_path = workspace_snapshot().get("latest_article")
    selected_path = resolve_preview_path(preview_articles, preferred or latest_path)
    return (
        preview_article_choices(preview_articles),
        selected_path,
        render_library_markdown(preview_articles, selected_path),
        render_article_markdown(preview_articles, selected_path),
        render_graph_html(preview_articles, selected_path),
    )


def _vault_outputs(preferred: str | None = None):
    choices, selected_path, library_md, browse_md, graph_html = _vault_state(preferred)
    return (
        library_md,
        gr.update(choices=choices, value=selected_path),
        browse_md,
        graph_html,
    )


def ingest_workflow(url: str, upload_path: str | None):
    url = (url or "").strip()
    if not url and not upload_path:
        raise gr.Error("Provide one source: either a URL or a PDF/Markdown upload.")
    if url and upload_path:
        raise gr.Error("Use one source at a time so the preview stays legible.")

    try:
        if url:
            result = ingest_demo_source(url, kind="url")
        else:
            result = ingest_demo_source(upload_path, kind="file")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    preferred = result.article_path.removeprefix("wiki/")
    library_md, picker_update, browse_md, graph_html = _vault_outputs(preferred)
    return (
        _status_markdown(),
        _article_markdown(result),
        _suggestions_markdown(result.suggestions, result.suggestion_mode),
        library_md,
        picker_update,
        browse_md,
        graph_html,
    )


def reset_workspace():
    seed_demo(reset=True)
    library_md, picker_update, browse_md, graph_html = _vault_outputs()
    return (
        _status_markdown(),
        "### New Page\n\n_Reset the preview workspace. Ingest a URL or upload a source to create a fresh page._",
        "### Next Questions\n\n_The starter corpus is back in place. Run one ingest to generate the next set of suggestions._",
        library_md,
        picker_update,
        browse_md,
        graph_html,
    )


def focus_article(selected_path: str):
    library_md, _, browse_md, graph_html = _vault_outputs(selected_path)
    return library_md, browse_md, graph_html


ensure_demo_workspace()
initial_choices, initial_selected_path, initial_library, initial_browse, initial_graph = _vault_state()

with gr.Blocks(title="Lore Preview") as demo:
    gr.Markdown(
        """
        <div class="hero">
          <div class="eyebrow">Lore Deterministic Preview</div>
          <h1>Ingest one source. Inspect the page. Browse the graph.</h1>
          <p>
            This Space is a zero-setup preview of Lore's wiki shape. It uses deterministic extraction
            and heuristics so you can inspect generated markdown, backlinks, and follow-up questions in
            seconds. For the full agent-maintained workflow, run Lore locally with Claude Code or MCP.
          </p>
        </div>
        """
    )
    gr.HTML(PREVIEW_NOTE_HTML)

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
                    ingest_button = gr.Button("Ingest Into Preview", variant="primary")
                    reset_button = gr.Button("Reset Preview Workspace")
                gr.Examples(SAMPLE_URLS, inputs=source_url, label="Sample URLs")

        with gr.Column(scale=4):
            with gr.Group(elem_classes=["panel"]):
                status = gr.Markdown(_status_markdown(), elem_classes=["mono"])
            with gr.Group(elem_classes=["panel"]):
                article_picker = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_selected_path,
                    label="Browse or focus a page",
                    info="The article picker also highlights the graph view.",
                )

    with gr.Tabs():
        with gr.Tab("Latest Run"):
            with gr.Row():
                article = gr.Markdown(
                    "### New Page\n\n_Ingest a source to write the next wiki page._",
                    elem_classes=["mono"],
                    height=720,
                )
                suggestions = gr.Markdown(
                    "### Next Questions\n\n_The preview will suggest what to explore next after each ingest._",
                    elem_classes=["mono"],
                )

        with gr.Tab("Browse The Vault"):
            with gr.Row():
                library = gr.Markdown(
                    initial_library,
                    elem_classes=["mono"],
                )
                browse_article = gr.Markdown(
                    initial_browse,
                    elem_classes=["mono"],
                    height=760,
                )

        with gr.Tab("Graph View"):
            graph = gr.HTML(initial_graph)

    ingest_button.click(
        ingest_workflow,
        inputs=[source_url, upload],
        outputs=[status, article, suggestions, library, article_picker, browse_article, graph],
    )
    reset_button.click(
        reset_workspace,
        outputs=[status, article, suggestions, library, article_picker, browse_article, graph],
    )
    article_picker.change(
        focus_article,
        inputs=[article_picker],
        outputs=[library, browse_article, graph],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
