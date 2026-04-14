"""Lore training CLI — curiosity training and model management."""

from __future__ import annotations

import json

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="lore-train", help="Curiosity training management", add_completion=False)
console = Console()


@app.command("curiosity")
def curiosity_cmd():
    """Train the model on your questioning patterns (SFT/GRPO)."""
    from lore.evolve.trainer import run_curiosity_training
    stats = run_curiosity_training()
    rprint(f"[green]✓[/green] Curiosity training complete: {stats}")


@app.command("status")
def status_cmd():
    """Show training status."""
    from lore.evolve.trajectory import get_question_trace_stats
    from lore.config import LORA_CHECKPOINTS_DIR, CURIOSITY_BOOTSTRAP_N

    q_stats = get_question_trace_stats()
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*")) if LORA_CHECKPOINTS_DIR.exists() else []
    total_traces = q_stats.get("total", 0)

    table = Table(title="Curiosity Training Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Question traces", str(total_traces))
    table.add_row("Untrained traces", str(q_stats.get("untrained", 0)))
    table.add_row("LoRA checkpoints", str(len(checkpoints)))
    table.add_row("Latest checkpoint",
                  checkpoints[-1].name if checkpoints else "none (base model)")
    table.add_row("Training mode",
                  "SFT (imitation)" if total_traces < CURIOSITY_BOOTSTRAP_N else "GRPO (RL)")

    console.print(table)

    from lore.evolve.trajectory import CURIOSITY_SUGGESTED_FLAG
    if CURIOSITY_SUGGESTED_FLAG.exists():
        rprint(
            f"\n[bold yellow]Ready to train:[/bold yellow] "
            f"{q_stats.get('untrained', 0)} new question traces.\n"
            f"  Run [bold]lore-train curiosity[/bold] to improve suggestions."
        )


@app.command("suggest")
def suggest_cmd(
    n: int = typer.Option(3, "--n", "-n", help="Number of suggestions"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Generate follow-up question suggestions based on your questioning style."""
    from lore.evolve.curiosity import generate_suggestions_with_mode

    suggestions, mode = generate_suggestions_with_mode(n=n)
    payload = {"mode": mode, "suggestions": suggestions}

    if json_output:
        print(json.dumps(payload))
        return

    if not suggestions:
        rprint("[yellow]No suggestions — wiki is empty or model not available.[/yellow]")
        return

    if mode == "heuristic":
        rprint("[yellow]Using heuristic fallback suggestions (no checkpoint yet).[/yellow]\n")
    elif mode == "daemon":
        rprint("[green]Using daemon-backed suggestions.[/green]\n")
    elif mode == "checkpoint":
        rprint("[green]Using local checkpoint suggestions.[/green]\n")

    rprint("\n[bold]Suggested follow-up questions:[/bold]\n")
    for i, s in enumerate(suggestions, 1):
        rprint(f"  [cyan]{i}.[/cyan] {s['question']}")
        rprint(f"     [dim]gap={s['gap_targeting']:.2f}  style={s['style_match']:.2f}  "
               f"novelty={s['novelty']:.2f}  specificity={s['specificity']:.2f}  "
               f"combined={s['combined']:.3f}[/dim]")
    rprint("")


@app.command("rollback")
def rollback_cmd(
    steps: int = typer.Option(1, "--steps", "-n", help="Number of checkpoints to roll back"),
):
    """Roll back to a previous LoRA checkpoint."""
    from lore.evolve.trainer import rollback_checkpoint
    rollback_checkpoint(steps)


@app.command("serve")
def serve_cmd(
    port: int = typer.Option(8765, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Start the daemon — keeps model in memory for instant suggestions."""
    import uvicorn
    from lore.evolve.daemon import app as daemon_app
    rprint(f"[green]Starting lore daemon on {host}:{port}[/green]")
    rprint("[dim]Keep this running in a terminal. Suggestions will be instant.[/dim]")
    uvicorn.run(daemon_app, host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
