"""Lore evolving agent CLI — manage RL training and model status."""

from __future__ import annotations

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="lore-train", help="Evolving agent training management", add_completion=False)
console = Console()


@app.command("train")
def train_cmd(
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
):
    """Trigger a GRPO/OPD training run on captured trajectories."""
    from lore.evolve.trainer import run_training
    stats = run_training(background=background)
    rprint(f"[green]✓[/green] Training complete: {stats}")


@app.command("status")
def status_cmd():
    """Show evolving agent training status."""
    from lore.evolve.trajectory import get_trajectory_stats
    from lore.evolve.buffer import get_reward_stats
    from lore.config import LORA_CHECKPOINTS_DIR
    from lore.evolve.distill import should_use_opd

    traj_stats = get_trajectory_stats()
    reward_stats = get_reward_stats()
    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*")) if LORA_CHECKPOINTS_DIR.exists() else []

    table = Table(title="Evolving Agent Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total trajectories", str(traj_stats.get("total", 0)))
    table.add_row("Untrained trajectories", str(traj_stats.get("untrained", 0)))
    table.add_row("Mean reward", f"{traj_stats.get('mean_reward', 0):.3f}")
    table.add_row("Reward std", f"{traj_stats.get('std_reward', 0):.3f}")
    table.add_row("Reward min/max",
                  f"{traj_stats.get('min_reward', 0):.3f} / {traj_stats.get('max_reward', 0):.3f}")
    table.add_row("LoRA checkpoints", str(len(checkpoints)))
    table.add_row("Latest checkpoint",
                  checkpoints[-1].name if checkpoints else "none (base model)")
    table.add_row("Training mode",
                  "OPD (distillation)" if should_use_opd() else "GRPO (RL)")

    if reward_stats:
        table.add_row("Recent mean (last 100)", f"{reward_stats.get('recent_mean', 0):.3f}")
        table.add_row("Recent std (last 100)", f"{reward_stats.get('recent_std', 0):.3f}")

    console.print(table)

    from lore.evolve.trajectory import TRAINING_SUGGESTED_FLAG, TRAIN_THRESHOLD
    untrained = traj_stats.get("untrained", 0)
    if TRAINING_SUGGESTED_FLAG.exists():
        rprint(
            f"\n[bold yellow]⚡ Ready to retrain:[/bold yellow] "
            f"{untrained} new trajectories available.\n"
            f"  Mode: [bold]{'OPD (distillation)' if should_use_opd() else 'GRPO (RL)'}[/bold]\n"
            f"  Run [bold]lore-train train[/bold] to proceed, or keep querying."
        )
    elif untrained > 0:
        remaining = TRAIN_THRESHOLD - untrained
        rprint(f"\n[dim]{remaining} more queries until retrain suggestion.[/dim]")


@app.command("rollback")
def rollback_cmd(
    steps: int = typer.Option(1, "--steps", "-n", help="Number of checkpoints to roll back"),
):
    """Roll back to a previous LoRA checkpoint."""
    from lore.config import LORA_CHECKPOINTS_DIR
    import shutil

    checkpoints = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    if len(checkpoints) <= steps:
        rprint(f"[red]Cannot roll back {steps} steps — only {len(checkpoints)} checkpoints exist[/red]")
        raise typer.Exit(1)

    to_delete = checkpoints[-steps:]
    for ckpt in to_delete:
        shutil.rmtree(str(ckpt))
        rprint(f"[yellow]Deleted:[/yellow] {ckpt.name}")

    remaining = sorted(LORA_CHECKPOINTS_DIR.glob("step-*"))
    rprint(f"[green]✓[/green] Rolled back to: {remaining[-1].name if remaining else 'base model'}")


@app.command("serve")
def serve_cmd(
    port: int = typer.Option(8765, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """Start the evolving agent inference server."""
    import uvicorn
    from lore.evolve.serve import app as fastapi_app
    rprint(f"[green]Starting evolving agent server on {host}:{port}[/green]")
    uvicorn.run(fastapi_app, host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
