"""Lore CLI — all /lore commands available from the terminal."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="lore", help="Personal ML research knowledge wiki", add_completion=False)
console = Console()


@app.command("ingest")
def ingest_cmd(
    source: str = typer.Argument(..., help="File path or URL to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already seen"),
):
    """Parse and fingerprint a new source document."""
    from lore.ingest.pipeline import ingest_file, ingest_url

    if source.startswith("http://") or source.startswith("https://"):
        chunks = ingest_url(source)
    else:
        chunks = ingest_file(source, force=force)

    if chunks:
        rprint(f"[green]✓[/green] Ingested {len(chunks)} chunks from: {source}")
        rprint("[yellow]Run 'lore-absorb' to compile new content into the wiki.[/yellow]")
    else:
        rprint(f"[yellow]Skipped (already ingested or empty): {source}[/yellow]")


@app.command("absorb")
def absorb_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Recompile all articles even if unchanged"),
):
    """Compile unprocessed raw sources into wiki articles."""
    from lore.compile.compiler import absorb
    stats = absorb(force=force)
    rprint(f"[green]✓[/green] Absorb complete:")
    for key, val in stats.items():
        rprint(f"  {key}: {val}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Number of results"),
    full: bool = typer.Option(False, "--full", help="Show full article content"),
):
    """Search the wiki with hybrid TF-IDF + embedding search."""
    from lore.index.search import search_and_read, format_search_results, hybrid_search

    if full:
        results = search_and_read(query, top_k=top_k)
        for r, content in results:
            rprint(f"\n[bold cyan]=== {r.title} ({r.article_path}) ===[/bold cyan]")
            rprint(content[:3000])
    else:
        results = hybrid_search(query, top_k=top_k)
        rprint(format_search_results(results))


@app.command("rebuild-index")
def rebuild_index_cmd(
    no_embeddings: bool = typer.Option(False, "--no-embeddings", help="Skip embedding index (faster)"),
):
    """Rebuild the TF-IDF and embedding search indexes."""
    from lore.index.store import rebuild_index
    stats = rebuild_index(use_embeddings=not no_embeddings)
    rprint(f"[green]✓[/green] Index rebuilt: {stats['articles']} articles")


@app.command("query")
def query_cmd(
    question: str = typer.Argument(..., help="Question to answer using the wiki"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    no_capture: bool = typer.Option(False, "--no-capture", help="Skip trajectory capture"),
):
    """Answer a question using the wiki (RAG + trajectory capture)."""
    from lore.query.agent import answer_question
    result = answer_question(
        question,
        top_k=top_k,
        capture_trajectory=not no_capture,
    )
    rprint(f"\n[bold]Q:[/bold] {question}\n")
    rprint(f"[bold]A:[/bold] {result.answer}\n")
    if result.output_path:
        rprint(f"[dim]Report: {result.output_path}[/dim]")


@app.command("health")
def health_cmd():
    """Audit the wiki for quality issues and undiscovered connections."""
    from lore.health.checker import run_health_check
    results = run_health_check()
    stats = results.get("stats", {})
    rprint(f"[green]✓[/green] Health check complete:")
    rprint(f"  Articles: {stats.get('total_articles', 0)}")
    rprint(f"  Broken links: {sum(len(v) for v in results.get('broken_links', {}).values())}")
    rprint(f"  Orphans: {len(results.get('orphans', []))}")
    rprint(f"  Stubs: {len(results.get('stubs', []))}")
    rprint(f"  Undiscovered connections: {len(results.get('connections', []))}")
    rprint("[dim]Full report: wiki/_meta/health_report.md[/dim]")


@app.command("cleanup")
def cleanup_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fixed without writing"),
):
    """Fix broken [[WikiLinks]], rebuild backlinks, report merged stubs."""
    from lore.compile.linker import find_broken_links, rebuild_all_backlinks, snap_wikilinks
    from lore.config import WIKI_DIR

    # 1. Find broken links
    broken = find_broken_links()
    if broken:
        rprint(f"[yellow]Fixing broken links in {len(broken)} articles...[/yellow]")
        for rel_path, bad_links in broken.items():
            full = WIKI_DIR / rel_path
            content = full.read_text(encoding="utf-8", errors="replace")
            fixed = snap_wikilinks(content)
            if fixed != content:
                if dry_run:
                    rprint(f"  [dim]would fix:[/dim] {rel_path}: {bad_links}")
                else:
                    full.write_text(fixed, encoding="utf-8")
                    rprint(f"  [green]fixed:[/green] {rel_path}")
    else:
        rprint("[green]✓[/green] No broken links.")

    # 2. Rebuild backlink footers
    if not dry_run:
        updated = rebuild_all_backlinks()
        rprint(f"[green]✓[/green] Backlinks rebuilt in {updated} articles.")

    # 3. Report stubs
    from lore.health.suggestions import suggest_new_articles
    stubs = suggest_new_articles()
    if stubs:
        rprint(f"\n[yellow]Stub candidates ({len(stubs)}) — concepts with no article:[/yellow]")
        for s in stubs[:10]:
            rprint(f"  [[{s}]]")


@app.command("reorganize")
def reorganize_cmd(
    dry_run: bool = typer.Option(True, "--dry-run/--apply",
                                  help="Propose moves (default) or apply them"),
):
    """Re-examine articles for taxonomy misclassification and propose rerouting."""
    import re, shutil
    from lore.config import WIKI_DIR, WIKI_CATEGORIES
    from lore.compile.taxonomy import classify_article
    from lore.index.store import rebuild_index

    moves: list[tuple] = []  # (current_path, proposed_category, article_title)

    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        if md_file.name.startswith("_") or any(p.name.startswith("_") for p in md_file.parents):
            continue
        current_cat = md_file.parent.name
        if current_cat not in WIKI_CATEGORIES:
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace")
        # Strip frontmatter for classification
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        proposed = classify_article(md_file.stem.replace("-", " ").title(), body[:400])
        if proposed != current_cat:
            moves.append((md_file, proposed, md_file.stem.replace("-", " ").title()))

    if not moves:
        rprint("[green]✓[/green] All articles are in the right category.")
        return

    rprint(f"[yellow]{len(moves)} reclassification suggestions:[/yellow]")
    for md_file, proposed, title in moves:
        current = md_file.parent.name
        rprint(f"  {title}: [red]{current}[/red] → [green]{proposed}[/green]")

    if dry_run:
        rprint("\n[dim]Run with --apply to move files.[/dim]")
        return

    # Apply moves
    for md_file, proposed, title in moves:
        dest_dir = WIKI_DIR / proposed
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / md_file.name
        shutil.move(str(md_file), str(dest))
        rprint(f"  [green]moved:[/green] {title} → {proposed}/")

    # Rebuild index after moves
    rebuild_index(use_embeddings=False)
    rprint("[green]✓[/green] Index rebuilt after reorganization.")


@app.command("status")
def status_cmd():
    """Show current knowledge base stats."""
    from lore.config import WIKI_DIR, DATA_DIR
    from lore.ingest.pipeline import get_ingestion_stats
    from lore.index.store import get_index_stats
    from lore.evolve.trajectory import get_trajectory_stats

    # Wiki article counts
    from collections import Counter
    category_counts: Counter = Counter()
    total_words = 0
    import re
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
            total_words += len(content.split())
            category_counts[md_file.parent.name] += 1
        except Exception:
            continue

    table = Table(title="Wiki Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total articles", str(sum(category_counts.values())))
    table.add_row("Total words", f"{total_words:,}")
    for cat, cnt in sorted(category_counts.items()):
        table.add_row(f"  {cat}", str(cnt))

    ingest_stats = get_ingestion_stats()
    table.add_row("Raw chunks", str(ingest_stats.get("total_chunks", 0)))
    table.add_row("Unabsorbed chunks", str(ingest_stats.get("unabsorbed", 0)))

    index_stats = get_index_stats()
    table.add_row("Index articles", str(index_stats.get("tfidf_articles", 0)))

    traj_stats = get_trajectory_stats()
    untrained = traj_stats.get("untrained", 0)
    table.add_row("Trajectories total", str(traj_stats.get("total", 0)))
    table.add_row("Untrained trajectories", str(untrained))
    table.add_row("Mean reward (all)", f"{traj_stats.get('mean_reward', 0):.3f}")

    console.print(table)

    # Surface training suggestion if flag is set
    from lore.evolve.trajectory import TRAINING_SUGGESTED_FLAG, TRAIN_THRESHOLD
    if TRAINING_SUGGESTED_FLAG.exists():
        rprint(
            f"\n[bold yellow]⚡ Retrain suggestion:[/bold yellow] "
            f"{untrained} new trajectories collected (threshold: {TRAIN_THRESHOLD}).\n"
            f"  Run [bold]lore-train train[/bold] to improve the LoRA model, "
            f"or keep querying to collect more data first."
        )


@app.command("render")
def render_cmd(
    output_type: str = typer.Argument(..., help="Output type: report | slides | charts"),
    topic: str = typer.Argument("", help="Topic for report/slides"),
):
    """Generate output artifacts (reports, slides, charts)."""
    if output_type == "report":
        if not topic:
            rprint("[red]Error: topic required for report[/red]")
            raise typer.Exit(1)
        from lore.render.report import generate_report
        path = generate_report(topic)
        rprint(f"[green]✓[/green] Report: {path}")

    elif output_type == "slides":
        if not topic:
            rprint("[red]Error: topic required for slides[/red]")
            raise typer.Exit(1)
        from lore.render.slides import generate_slides
        path = generate_slides(topic)
        rprint(f"[green]✓[/green] Slides: {path}")

    elif output_type == "charts":
        from lore.render.charts import (
            plot_category_distribution,
            plot_knowledge_growth,
            plot_backlink_graph,
            plot_reward_history,
        )
        for fn in [plot_category_distribution, plot_knowledge_growth,
                   plot_backlink_graph, plot_reward_history]:
            try:
                path = fn(save=True)
                if path:
                    rprint(f"[green]✓[/green] {path}")
            except Exception as e:
                rprint(f"[yellow]warn[/yellow] {fn.__name__}: {e}")

    else:
        rprint(f"[red]Unknown output type: {output_type}[/red]")
        rprint("Valid: report | slides | charts")


def main():
    app()


# Individual entry point wrappers (each spins up the full app with sys.argv)
def ingest_main():
    import sys
    sys.argv = ["lore", "ingest"] + sys.argv[1:]
    app()


def search_main():
    import sys
    sys.argv = ["lore", "search"] + sys.argv[1:]
    app()


def absorb_main():
    import sys
    sys.argv = ["lore", "absorb"] + sys.argv[1:]
    app()


def rebuild_index_main():
    import sys
    sys.argv = ["lore", "rebuild-index"] + sys.argv[1:]
    app()


def health_main():
    import sys
    sys.argv = ["lore", "health"] + sys.argv[1:]
    app()


def status_main():
    import sys
    sys.argv = ["lore", "status"] + sys.argv[1:]
    app()


def render_main():
    import sys
    sys.argv = ["lore", "render"] + sys.argv[1:]
    app()


def query_main():
    import sys
    sys.argv = ["lore", "query"] + sys.argv[1:]
    app()


def cleanup_main():
    import sys
    sys.argv = ["lore", "cleanup"] + sys.argv[1:]
    app()


def reorganize_main():
    import sys
    sys.argv = ["lore", "reorganize"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    main()
