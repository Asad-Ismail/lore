"""
Lore web UI — FastAPI backend + single-page frontend.

Run:  lore-web            (default port 7860)
      lore-web --port 8080
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from lore.config import WIKI_DIR, DATA_DIR, OUTPUTS_DIR

app = FastAPI(title="Lore", docs_url=None, redoc_url=None)


# ── API schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    try:
        from lore.evolve.trajectory import get_trajectory_stats, TRAINING_SUGGESTED_FLAG
        from lore.evolve.buffer import get_reward_stats
        from lore.index.store import get_index_stats

        traj = get_trajectory_stats()
        rewards = get_reward_stats()
        index = get_index_stats()

        # Latest LoRA checkpoint
        ckpt_dir = DATA_DIR / "lora_checkpoints"
        checkpoints = sorted(ckpt_dir.glob("step-*")) if ckpt_dir.exists() else []
        latest_ckpt = checkpoints[-1].name if checkpoints else "base model"

        # Article counts
        article_counts: dict[str, int] = {}
        if WIKI_DIR.exists():
            for cat_dir in sorted(WIKI_DIR.iterdir()):
                if cat_dir.is_dir() and not cat_dir.name.startswith("_"):
                    article_counts[cat_dir.name] = len(list(cat_dir.glob("*.md")))

        return {
            "articles_total": sum(article_counts.values()),
            "article_counts": article_counts,
            "trajectories_total": traj.get("total", 0),
            "trajectories_untrained": traj.get("untrained", 0),
            "mean_reward": rewards.get("recent_mean") or rewards.get("all_mean", 0.0),
            "indexed_articles": index.get("tfidf_articles") or index.get("embedding_articles", 0),
            "latest_checkpoint": latest_ckpt,
            "retrain_suggested": TRAINING_SUGGESTED_FLAG.exists(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/articles")
async def list_articles():
    """Return all articles grouped by category."""
    if not WIKI_DIR.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for cat_dir in sorted(WIKI_DIR.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue
        articles = []
        for md in sorted(cat_dir.glob("*.md")):
            title = md.stem.replace("-", " ").title()
            # Try to read frontmatter title
            try:
                first = md.read_text(encoding="utf-8", errors="replace")[:300]
                for line in first.splitlines():
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip()
                        break
            except Exception:
                pass
            articles.append({"title": title, "path": str(md.relative_to(WIKI_DIR))})
        if articles:
            result[cat_dir.name] = articles
    return result


@app.get("/api/article")
async def get_article(path: str = Query(...)):
    """Return markdown content of an article."""
    safe = (WIKI_DIR / path).resolve()
    if not str(safe).startswith(str(WIKI_DIR.resolve())):
        raise HTTPException(400, "Invalid path")
    if not safe.exists():
        raise HTTPException(404, "Not found")
    return {"content": safe.read_text(encoding="utf-8", errors="replace"), "path": path}


@app.get("/api/search")
async def search(q: str = Query(...), top_k: int = 8):
    try:
        from lore.index.search import hybrid_search
        results = hybrid_search(q, top_k=top_k)
        return [
            {"title": r.title, "path": r.article_path, "score": round(r.score, 4),
             "snippet": r.snippet[:200] if hasattr(r, "snippet") else ""}
            for r in results
        ]
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        from lore.query.agent import answer_question
        result = answer_question(req.question, top_k=req.top_k, save_output=True)
        return {
            "answer": result.answer,
            "retrieved": result.retrieved_paths,
            "citations": result.citations,
            "citation_validation": result.citation_validation,
            "output_path": result.output_path,
            "reward": _extract_reward(result),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health-report")
async def health_report():
    report_path = WIKI_DIR / "_meta" / "health_report.md"
    if not report_path.exists():
        return {"content": "_No health report yet. Run `lore health` to generate one._"}
    return {"content": report_path.read_text(encoding="utf-8", errors="replace")}


def _extract_reward(result) -> dict:
    """Pull reward breakdown out of a trajectory if one was saved."""
    try:
        from lore.evolve.trajectory import load_latest_trajectory
        traj = load_latest_trajectory()
        if traj:
            m = traj.metadata or {}
            coverage = traj.coverage_score if traj.coverage_score >= 0 else 0.0
            return {
                "total": round(traj.reward, 3),
                "grounding": round(m.get("grounding", 0.0), 3),
                "citation": round(m.get("citation", 0.0), 3),
                "coverage": round(coverage, 3),
                "fluency": round(m.get("fluency", 0.0), 3),
            }
    except Exception:
        pass
    return {}


# ── HTML UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lore web UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n  Lore  →  http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ── Embedded HTML/CSS/JS ──────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lore</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
/* ── Reset & base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #0c0e10;
  --bg2:        #111418;
  --bg3:        #181c21;
  --border:     #252a30;
  --border2:    #2e343c;
  --text:       #ddd0b0;
  --text2:      #9a8f7a;
  --text3:      #5a5248;
  --gold:       #c49a3c;
  --gold2:      #e8b84b;
  --teal:       #2d9a8e;
  --teal2:      #3bbdaf;
  --blue:       #5b9bd5;
  --red:        #c4503c;
  --font-mono:  'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  --font-serif: 'Lora', 'Georgia', serif;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
  overflow: hidden;
}

/* subtle grain */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 9999;
  opacity: 0.4;
}

/* ── Layout ───────────────────────────────────────────────────────────── */
#app {
  display: grid;
  grid-template-rows: 38px 1fr;
  grid-template-columns: 240px 1fr;
  height: 100vh;
}

/* ── Top bar ──────────────────────────────────────────────────────────── */
#topbar {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 0 16px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  letter-spacing: 0.08em;
}
#logo {
  font-weight: 500;
  font-size: 13px;
  letter-spacing: 0.15em;
  color: var(--gold);
  text-transform: uppercase;
  flex-shrink: 0;
}
.stat { color: var(--text2); }
.stat b { color: var(--text); font-weight: 400; }
.stat-sep { color: var(--border2); }
#retrain-badge {
  margin-left: auto;
  display: none;
  padding: 2px 8px;
  background: rgba(196,154,60,0.12);
  border: 1px solid rgba(196,154,60,0.3);
  color: var(--gold);
  font-size: 10px;
  letter-spacing: 0.06em;
  cursor: pointer;
}
#retrain-badge.visible { display: block; }
#retrain-badge:hover { background: rgba(196,154,60,0.2); }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
#sidebar {
  display: flex;
  flex-direction: column;
  background: var(--bg2);
  border-right: 1px solid var(--border);
  overflow: hidden;
}

#search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
#search-box input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  caret-color: var(--gold);
}
#search-box input::placeholder { color: var(--text3); }
#search-box .icon { color: var(--text3); font-size: 11px; }

#article-tree {
  flex: 1;
  overflow-y: auto;
  padding: 6px 0;
  scrollbar-width: thin;
  scrollbar-color: var(--border2) transparent;
}
#article-tree::-webkit-scrollbar { width: 4px; }
#article-tree::-webkit-scrollbar-thumb { background: var(--border2); }

.cat-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  cursor: pointer;
  color: var(--text2);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  user-select: none;
}
.cat-header:hover { color: var(--text); }
.cat-arrow { color: var(--text3); transition: transform 0.15s; font-size: 9px; }
.cat-arrow.open { transform: rotate(90deg); }
.cat-count { margin-left: auto; color: var(--text3); font-size: 10px; }

.cat-items { display: none; }
.cat-items.open { display: block; }

.article-item {
  padding: 3px 12px 3px 24px;
  cursor: pointer;
  color: var(--text2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 12px;
  transition: color 0.1s, background 0.1s;
}
.article-item:hover { color: var(--text); background: rgba(255,255,255,0.03); }
.article-item.active { color: var(--gold); }

#sidebar-stats {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  padding: 10px 12px;
}
.sstat {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  margin-bottom: 3px;
  color: var(--text3);
}
.sstat b { color: var(--text2); font-weight: 400; }

/* ── Main panel ───────────────────────────────────────────────────────── */
#main {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Tabs */
#tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  background: var(--bg2);
}
.tab {
  padding: 8px 16px;
  cursor: pointer;
  color: var(--text3);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color 0.15s;
}
.tab:hover { color: var(--text2); }
.tab.active { color: var(--gold); border-bottom-color: var(--gold); }

/* Panels */
.panel { display: none; flex: 1; overflow: hidden; flex-direction: column; }
.panel.active { display: flex; }

/* ── Query panel ──────────────────────────────────────────────────────── */
#query-input-wrap {
  padding: 20px 24px 16px;
  flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
#query-label {
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 8px;
}
#query-wrap {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
#query-input {
  flex: 1;
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 0;
  outline: none;
  padding: 10px 14px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 13px;
  resize: none;
  line-height: 1.5;
  caret-color: var(--gold);
  transition: border-color 0.15s;
  min-height: 44px;
  max-height: 120px;
}
#query-input:focus { border-color: var(--gold); }
#query-input::placeholder { color: var(--text3); }
#query-btn {
  flex-shrink: 0;
  padding: 10px 18px;
  background: var(--gold);
  color: var(--bg);
  border: none;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: background 0.15s;
  height: 44px;
}
#query-btn:hover { background: var(--gold2); }
#query-btn:disabled { opacity: 0.5; cursor: not-allowed; }

#query-output {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  scrollbar-width: thin;
  scrollbar-color: var(--border2) transparent;
}
#query-output::-webkit-scrollbar { width: 4px; }
#query-output::-webkit-scrollbar-thumb { background: var(--border2); }

#answer-placeholder {
  color: var(--text3);
  font-size: 12px;
  padding-top: 8px;
  font-style: italic;
}

.answer-block { display: none; }
.answer-block.visible { display: block; }

.answer-section-label {
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.answer-section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

#answer-text {
  font-family: var(--font-serif);
  font-size: 14px;
  line-height: 1.8;
  color: var(--text);
  margin-bottom: 24px;
  white-space: pre-wrap;
}

/* WikiLinks in answers */
#answer-text .wikilink {
  color: var(--blue);
  border-bottom: 1px dotted rgba(91,155,213,0.4);
  cursor: pointer;
  text-decoration: none;
}
#answer-text .wikilink:hover { color: #7ec0f0; border-bottom-style: solid; }

/* Reward breakdown */
#reward-row {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.reward-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  font-size: 11px;
}
.reward-pill .label { color: var(--text3); }
.reward-pill .val { color: var(--teal2); font-weight: 500; }
.reward-pill.total { border-color: var(--teal); }
.reward-pill.total .val { color: var(--teal2); font-size: 12px; }

/* Sources */
#sources-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 20px;
}
.source-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 0;
  cursor: pointer;
  color: var(--text2);
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-row:hover { color: var(--text); }
.source-dot { width: 6px; height: 6px; background: var(--teal); flex-shrink: 0; }
.source-path { color: var(--text3); font-size: 10px; margin-left: auto; }

/* Citation validation */
#citations-list { display: flex; flex-direction: column; gap: 3px; margin-bottom: 20px; }
.cite-row { display: flex; align-items: center; gap: 8px; font-size: 11px; padding: 2px 0; }
.cite-badge {
  padding: 1px 6px; font-size: 10px; letter-spacing: 0.05em;
  flex-shrink: 0;
}
.cite-badge.valid   { background: rgba(45,154,142,0.15); color: var(--teal2); border: 1px solid rgba(45,154,142,0.3); }
.cite-badge.halluc  { background: rgba(196,80,60,0.15);  color: var(--red);   border: 1px solid rgba(196,80,60,0.3); }
.cite-badge.missing { background: rgba(90,82,72,0.2);    color: var(--text3); border: 1px solid var(--border2); }
.cite-name { color: var(--text2); cursor: pointer; }
.cite-name:hover { color: var(--blue); }

/* ── Article panel ────────────────────────────────────────────────────── */
#article-panel { padding: 0; }
#article-header {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
#article-title {
  font-family: var(--font-serif);
  font-size: 20px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}
#article-meta { font-size: 11px; color: var(--text3); }
#article-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  font-family: var(--font-serif);
  font-size: 14px;
  line-height: 1.85;
  color: var(--text);
  scrollbar-width: thin;
  scrollbar-color: var(--border2) transparent;
}
#article-body::-webkit-scrollbar { width: 4px; }
#article-body::-webkit-scrollbar-thumb { background: var(--border2); }
#article-body h1, #article-body h2, #article-body h3 {
  font-family: var(--font-mono);
  font-weight: 500;
  color: var(--text);
  margin: 1.2em 0 0.5em;
  letter-spacing: 0.04em;
}
#article-body h1 { font-size: 18px; color: var(--gold); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
#article-body h2 { font-size: 14px; color: var(--text2); }
#article-body h3 { font-size: 13px; color: var(--text3); }
#article-body p { margin-bottom: 1em; }
#article-body ul, #article-body ol { padding-left: 1.5em; margin-bottom: 1em; }
#article-body li { margin-bottom: 0.25em; }
#article-body code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg3);
  padding: 1px 5px;
  border: 1px solid var(--border);
  color: var(--gold);
}
#article-body pre {
  background: var(--bg3);
  border: 1px solid var(--border);
  padding: 12px 16px;
  overflow-x: auto;
  margin-bottom: 1em;
}
#article-body pre code { background: none; border: none; padding: 0; color: var(--text2); }
#article-placeholder {
  color: var(--text3);
  font-family: var(--font-mono);
  font-size: 12px;
  font-style: italic;
  padding: 20px 24px;
}
.wikilink-inline {
  color: var(--blue);
  border-bottom: 1px dotted rgba(91,155,213,0.4);
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 0.9em;
}
.wikilink-inline:hover { color: #7ec0f0; }

/* ── Search panel ────────────────────────────────────────────────────── */
#search-panel { padding: 0; }
#search-panel-inner { padding: 20px 24px; flex: 1; overflow-y: auto; }
#search-main-wrap {
  display: flex; gap: 10px; margin-bottom: 20px;
}
#search-main-input {
  flex: 1;
  background: var(--bg3);
  border: 1px solid var(--border2);
  outline: none;
  padding: 9px 14px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 13px;
  caret-color: var(--gold);
  transition: border-color 0.15s;
}
#search-main-input:focus { border-color: var(--gold); }
#search-main-btn {
  padding: 9px 16px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  color: var(--text2);
  font-family: var(--font-mono);
  font-size: 11px;
  cursor: pointer;
  letter-spacing: 0.08em;
  transition: border-color 0.15s, color 0.15s;
}
#search-main-btn:hover { border-color: var(--gold); color: var(--gold); }
.search-result {
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
}
.search-result:last-child { border-bottom: none; }
.search-result:hover .sr-title { color: var(--gold); }
.sr-title { font-size: 14px; color: var(--text); margin-bottom: 3px; transition: color 0.1s; }
.sr-path { font-size: 10px; color: var(--text3); margin-bottom: 5px; }
.sr-snippet { font-size: 12px; color: var(--text2); font-family: var(--font-serif); line-height: 1.5; }
.sr-score { float: right; font-size: 10px; color: var(--teal); }
#search-empty { color: var(--text3); font-size: 12px; font-style: italic; }

/* ── Spinner ──────────────────────────────────────────────────────────── */
.spinner {
  display: inline-block;
  width: 12px; height: 12px;
  border: 1px solid var(--border2);
  border-top-color: var(--gold);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Toast ────────────────────────────────────────────────────────────── */
#toast {
  position: fixed; bottom: 16px; right: 16px;
  background: var(--bg3); border: 1px solid var(--border2);
  padding: 8px 14px; font-size: 11px; color: var(--text2);
  display: none; z-index: 1000;
}
#toast.show { display: block; }
</style>
</head>
<body>
<div id="app">

  <!-- Top bar -->
  <div id="topbar">
    <span id="logo">▸ Lore</span>
    <span class="stat"><b id="stat-articles">—</b> articles</span>
    <span class="stat-sep">·</span>
    <span class="stat"><b id="stat-traj">—</b> trajectories</span>
    <span class="stat-sep">·</span>
    <span class="stat">LoRA <b id="stat-ckpt">—</b></span>
    <span class="stat-sep">·</span>
    <span class="stat">reward <b id="stat-reward">—</b></span>
    <span id="retrain-badge" title="Run lore-train train">⚡ retrain ready</span>
  </div>

  <!-- Sidebar -->
  <div id="sidebar">
    <div id="search-box">
      <span class="icon">⌕</span>
      <input id="sidebar-search" type="text" placeholder="filter articles…" autocomplete="off">
    </div>
    <div id="article-tree"></div>
    <div id="sidebar-stats">
      <div class="sstat"><span>indexed</span><b id="ss-indexed">—</b></div>
      <div class="sstat"><span>untrained</span><b id="ss-untrained">—</b></div>
    </div>
  </div>

  <!-- Main -->
  <div id="main">
    <div id="tabs">
      <div class="tab active" data-tab="query">Query</div>
      <div class="tab" data-tab="article">Article</div>
      <div class="tab" data-tab="search">Search</div>
    </div>

    <!-- Query panel -->
    <div class="panel active" id="query-panel">
      <div id="query-input-wrap">
        <div id="query-label">Ask your wiki</div>
        <div id="query-wrap">
          <textarea id="query-input" rows="1" placeholder="What is the key tradeoff between GPTQ and AWQ?"></textarea>
          <button id="query-btn">Ask</button>
        </div>
      </div>
      <div id="query-output">
        <div id="answer-placeholder">Type a question and press Ask — or ⌘+Enter.</div>
        <div id="answer-block" class="answer-block">
          <div class="answer-section-label">Answer</div>
          <div id="answer-text"></div>
          <div id="reward-row"></div>
          <div class="answer-section-label">Sources</div>
          <div id="sources-list"></div>
          <div id="citations-section" style="display:none">
            <div class="answer-section-label">Citations</div>
            <div id="citations-list"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Article panel -->
    <div class="panel" id="article-panel">
      <div id="article-placeholder">Select an article from the sidebar.</div>
      <div id="article-content" style="display:none; flex:1; flex-direction:column; overflow:hidden;">
        <div id="article-header">
          <div id="article-title"></div>
          <div id="article-meta"></div>
        </div>
        <div id="article-body"></div>
      </div>
    </div>

    <!-- Search panel -->
    <div class="panel" id="search-panel">
      <div id="search-panel-inner">
        <div id="search-main-wrap">
          <input id="search-main-input" type="text" placeholder="Search the wiki…" autocomplete="off">
          <button id="search-main-btn">Search</button>
        </div>
        <div id="search-results"><div id="search-empty">Enter a query above.</div></div>
      </div>
    </div>

  </div><!-- /main -->
</div><!-- /app -->
<div id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const state = { articles: {}, currentArticlePath: null };

// ── Utilities ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const toast = (msg, dur=2500) => {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), dur);
};

async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await Promise.all([loadStatus(), loadArticles()]);
}

// ── Status ─────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const s = await api('/api/status');
    $('stat-articles').textContent = s.articles_total ?? '—';
    $('stat-traj').textContent = s.trajectories_total ?? '—';
    $('stat-ckpt').textContent = s.latest_checkpoint ?? '—';
    $('stat-reward').textContent = s.mean_reward != null ? s.mean_reward.toFixed(3) : '—';
    $('ss-indexed').textContent = s.indexed_articles ?? '—';
    $('ss-untrained').textContent = s.trajectories_untrained ?? '—';
    if (s.retrain_suggested) $('retrain-badge').classList.add('visible');
  } catch(e) { console.warn('status:', e); }
}

// ── Article tree ───────────────────────────────────────────────────────────
async function loadArticles() {
  try {
    state.articles = await api('/api/articles');
    renderTree(state.articles);
  } catch(e) { console.warn('articles:', e); }
}

function renderTree(articles, filter='') {
  const tree = $('article-tree');
  tree.innerHTML = '';
  const fl = filter.toLowerCase();

  for (const [cat, items] of Object.entries(articles)) {
    const filtered = filter ? items.filter(a => a.title.toLowerCase().includes(fl)) : items;
    if (!filtered.length) continue;

    const header = document.createElement('div');
    header.className = 'cat-header';
    header.innerHTML = `<span class="cat-arrow open">▶</span>
      <span>${cat}</span>
      <span class="cat-count">${filtered.length}</span>`;
    tree.appendChild(header);

    const list = document.createElement('div');
    list.className = 'cat-items open';
    filtered.forEach(a => {
      const item = document.createElement('div');
      item.className = 'article-item' + (a.path === state.currentArticlePath ? ' active' : '');
      item.textContent = a.title;
      item.dataset.path = a.path;
      item.addEventListener('click', () => openArticle(a.path, a.title));
      list.appendChild(item);
    });
    tree.appendChild(list);

    header.addEventListener('click', () => {
      list.classList.toggle('open');
      header.querySelector('.cat-arrow').classList.toggle('open');
    });
  }
}

$('sidebar-search').addEventListener('input', e => renderTree(state.articles, e.target.value));

// ── Article view ───────────────────────────────────────────────────────────
async function openArticle(path, title) {
  state.currentArticlePath = path;
  switchTab('article');
  renderTree(state.articles, $('sidebar-search').value);

  $('article-placeholder').style.display = 'none';
  $('article-content').style.display = 'flex';
  $('article-title').textContent = title || path;
  $('article-meta').textContent = 'wiki/' + path;
  $('article-body').innerHTML = '<span class="spinner"></span>';

  try {
    const data = await api(`/api/article?path=${encodeURIComponent(path)}`);
    $('article-body').innerHTML = renderMarkdown(data.content);
    // Make wikilinks clickable
    $('article-body').querySelectorAll('.wikilink-inline').forEach(el => {
      el.addEventListener('click', () => {
        const slug = el.dataset.slug;
        const found = findArticleBySlug(slug);
        if (found) openArticle(found.path, found.title);
        else toast(`Article "${slug}" not in index`);
      });
    });
  } catch(e) {
    $('article-body').textContent = 'Error loading article: ' + e.message;
  }
}

function findArticleBySlug(slug) {
  const sl = slug.toLowerCase().replace(/\s+/g, '-');
  for (const items of Object.values(state.articles)) {
    for (const a of items) {
      if (a.path.toLowerCase().includes(sl) || a.title.toLowerCase() === slug.toLowerCase())
        return a;
    }
  }
  return null;
}

// ── Minimal markdown renderer ──────────────────────────────────────────────
function renderMarkdown(md) {
  // Strip YAML frontmatter
  md = md.replace(/^---[\s\S]*?---\n/, '');

  // Escape HTML
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  // Code blocks
  md = md.replace(/```[\w]*\n([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${esc(code.trim())}</code></pre>`);

  // Headers
  md = md.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  md = md.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold / italic
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // WikiLinks [[Title]]
  md = md.replace(/\[\[([^\]|#]+?)(?:\|([^\]]+))?\]\]/g, (_, slug, label) =>
    `<span class="wikilink-inline" data-slug="${esc(slug)}" title="Open: ${esc(slug)}">${esc(label || slug)}</span>`);

  // Inline code
  md = md.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Lists
  md = md.replace(/^- (.+)$/gm, '<li>$1</li>');
  md = md.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);

  // Paragraphs (wrap non-tag lines)
  const lines = md.split('\n');
  const out = [];
  let inPre = false;
  for (const line of lines) {
    if (line.startsWith('<pre')) inPre = true;
    if (line.startsWith('</pre')) inPre = false;
    if (!inPre && line.trim() && !line.match(/^<[hupol]/)) {
      out.push(`<p>${line}</p>`);
    } else {
      out.push(line);
    }
  }
  return out.join('\n');
}

// ── Query ──────────────────────────────────────────────────────────────────
async function runQuery() {
  const q = $('query-input').value.trim();
  if (!q) return;

  $('query-btn').disabled = true;
  $('query-btn').innerHTML = '<span class="spinner"></span>';
  $('answer-placeholder').style.display = 'none';
  $('answer-block').classList.remove('visible');

  try {
    const data = await api('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 8 }),
    });

    // Answer text with wikilink highlighting
    $('answer-text').innerHTML = highlightWikilinks(data.answer);
    $('answer-text').querySelectorAll('.wikilink').forEach(el => {
      el.addEventListener('click', () => {
        const found = findArticleBySlug(el.dataset.slug);
        if (found) openArticle(found.path, found.title);
        else toast(`No article for "${el.dataset.slug}"`);
      });
    });

    // Reward pills
    const rw = data.reward || {};
    const pillsHtml = rw.total != null ? `
      <div class="reward-pill total"><span class="label">total</span><span class="val">${rw.total.toFixed(3)}</span></div>
      <div class="reward-pill"><span class="label">grounding</span><span class="val">${rw.grounding?.toFixed(3) ?? '—'}</span></div>
      <div class="reward-pill"><span class="label">citation</span><span class="val">${rw.citation?.toFixed(3) ?? '—'}</span></div>
      <div class="reward-pill"><span class="label">coverage</span><span class="val">${rw.coverage?.toFixed(3) ?? '—'}</span></div>
      <div class="reward-pill"><span class="label">fluency</span><span class="val">${rw.fluency?.toFixed(3) ?? '—'}</span></div>
    ` : '';
    $('reward-row').innerHTML = pillsHtml;

    // Sources
    const srcHtml = (data.retrieved || []).map(p => {
      const name = p.split('/').pop().replace('.md', '').replace(/-/g, ' ');
      return `<div class="source-row" data-path="${p}">
        <span class="source-dot"></span>
        <span>${name}</span>
        <span class="source-path">${p}</span>
      </div>`;
    }).join('');
    $('sources-list').innerHTML = srcHtml || '<span style="color:var(--text3);font-size:12px">No sources retrieved.</span>';
    $('sources-list').querySelectorAll('.source-row[data-path]').forEach(el => {
      el.addEventListener('click', () => {
        const found = findArticleBySlug(el.dataset.path.split('/').pop().replace('.md',''));
        if (found) openArticle(found.path, found.title);
      });
    });

    // Citation validation
    const cv = data.citation_validation || {};
    const valid = new Set(cv.valid || []);
    const halluc = new Set(cv.hallucinated || []);
    const missing = new Set(cv.nonexistent || []);
    const allCites = [...valid, ...halluc, ...missing];
    if (allCites.length) {
      $('citations-section').style.display = 'block';
      $('citations-list').innerHTML = allCites.map(c => {
        const badge = valid.has(c) ? ['valid','valid'] : halluc.has(c) ? ['halluc','hallucinated'] : ['missing','not found'];
        return `<div class="cite-row">
          <span class="cite-badge ${badge[0]}">${badge[1]}</span>
          <span class="cite-name" data-slug="${c}">[[${c}]]</span>
        </div>`;
      }).join('');
      $('citations-list').querySelectorAll('.cite-name').forEach(el => {
        el.addEventListener('click', () => {
          const found = findArticleBySlug(el.dataset.slug);
          if (found) openArticle(found.path, found.title);
        });
      });
    } else {
      $('citations-section').style.display = 'none';
    }

    $('answer-block').classList.add('visible');
    loadStatus();
  } catch(e) {
    $('answer-placeholder').textContent = 'Error: ' + e.message;
    $('answer-placeholder').style.display = 'block';
  } finally {
    $('query-btn').disabled = false;
    $('query-btn').textContent = 'Ask';
  }
}

function highlightWikilinks(text) {
  return text.replace(/\[\[([^\]|#]+?)(?:\|([^\]]+))?\]\]/g, (_, slug, label) =>
    `<a class="wikilink" data-slug="${slug}" href="#">${label || slug}</a>`);
}

$('query-btn').addEventListener('click', runQuery);
$('query-input').addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runQuery(); }
});
// Auto-resize textarea
$('query-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ── Search ─────────────────────────────────────────────────────────────────
async function runSearch() {
  const q = $('search-main-input').value.trim();
  if (!q) return;
  $('search-results').innerHTML = '<span class="spinner"></span>';
  try {
    const results = await api(`/api/search?q=${encodeURIComponent(q)}&top_k=15`);
    if (!results.length) {
      $('search-results').innerHTML = '<div id="search-empty">No results.</div>';
      return;
    }
    $('search-results').innerHTML = results.map(r => `
      <div class="search-result" data-path="${r.path}">
        <div class="sr-title">${r.title} <span class="sr-score">${r.score.toFixed(4)}</span></div>
        <div class="sr-path">wiki/${r.path}</div>
        <div class="sr-snippet">${r.snippet}</div>
      </div>
    `).join('');
    $('search-results').querySelectorAll('.search-result').forEach(el => {
      el.addEventListener('click', () => {
        const path = el.dataset.path;
        const title = el.querySelector('.sr-title').textContent.trim();
        openArticle(path, title);
      });
    });
  } catch(e) {
    $('search-results').innerHTML = `<div style="color:var(--red);font-size:12px">${e.message}</div>`;
  }
}

$('search-main-btn').addEventListener('click', runSearch);
$('search-main-input').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

// ── Tabs ───────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === name + '-panel'));
}
document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => switchTab(t.dataset.tab)));

// ── Start ──────────────────────────────────────────────────────────────────
init();
setInterval(loadStatus, 30000);
</script>
</body>
</html>"""
