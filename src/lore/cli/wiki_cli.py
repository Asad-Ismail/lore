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
        rprint("[yellow]Source saved to raw/. Tell your agent to ingest it.[/yellow]")
    else:
        rprint(f"[yellow]Skipped (already ingested or empty): {source}[/yellow]")



@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Number of results"),
    full: bool = typer.Option(False, "--full", help="Show full article content"),
):
    """Search the wiki with TF-IDF search."""
    from lore.index.search import search_and_read, format_search_results, search_wiki

    if full:
        results = search_and_read(query, top_k=top_k)
        for r, content in results:
            rprint(f"\n[bold cyan]=== {r.title} ({r.article_path}) ===[/bold cyan]")
            rprint(content[:3000])
    else:
        results = search_wiki(query, top_k=top_k)
        rprint(format_search_results(results))


@app.command("rebuild-index")
def rebuild_index_cmd():
    """Rebuild the TF-IDF search index."""
    from lore.index.store import rebuild_index
    stats = rebuild_index()
    rprint(f"[green]✓[/green] Index rebuilt: {stats['articles']} articles")



@app.command("trace")
def trace_cmd(
    question: str = typer.Argument(..., help="The question to record"),
):
    """Capture a question trace for curiosity training (no GPU needed)."""
    from lore.evolve.curiosity import build_wiki_state_summary
    from lore.evolve.trajectory import capture_question_trace
    wiki_state = build_wiki_state_summary()
    trace = capture_question_trace(question, wiki_state)
    rprint(f"[green]✓[/green] Question trace saved ({trace.id[:8]})")


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
    from lore.linker import find_broken_links, rebuild_all_backlinks, snap_wikilinks
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




@app.command("status")
def status_cmd():
    """Show current knowledge base stats."""
    from lore.config import WIKI_DIR, DATA_DIR
    from lore.ingest.pipeline import get_ingestion_stats

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
    table.add_row("Ingested sources", str(ingest_stats.get("sources", 0)))

    from lore.evolve.trajectory import get_question_trace_stats, CURIOSITY_SUGGESTED_FLAG
    q_stats = get_question_trace_stats()
    table.add_row("Question traces", str(q_stats.get("total", 0)))
    table.add_row("Untrained traces", str(q_stats.get("untrained", 0)))

    console.print(table)

    if CURIOSITY_SUGGESTED_FLAG.exists():
        rprint(
            f"\n[bold yellow]Ready to train:[/bold yellow] "
            f"{q_stats.get('untrained', 0)} new question traces.\n"
            f"  Run [bold]lore-train curiosity[/bold] to improve suggestions."
        )




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




def cleanup_main():
    import sys
    sys.argv = ["lore", "cleanup"] + sys.argv[1:]
    app()


def trace_main():
    import sys
    sys.argv = ["lore", "trace"] + sys.argv[1:]
    app()



if __name__ == "__main__":
    main()
