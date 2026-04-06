"""Generate matplotlib charts and visualizations from wiki data."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lore.config import WIKI_DIR, OUTPUTS_DIR


def _get_charts_dir() -> Path:
    d = OUTPUTS_DIR / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_category_distribution(save: bool = True) -> Path | None:
    """Bar chart: article count per wiki category."""
    import matplotlib.pyplot as plt

    counts: Counter = Counter()
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        counts[md_file.parent.name] += 1

    if not counts:
        print("[charts] No wiki articles found.")
        return None

    categories = sorted(counts.keys())
    values = [counts[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, values, color="steelblue", alpha=0.8)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Category")
    ax.set_ylabel("Article Count")
    ax.set_title("Wiki Articles by Category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        path = _get_charts_dir() / f"category-dist-{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[charts] Saved: {path}")
        return path
    plt.show()
    return None


def plot_knowledge_growth(save: bool = True) -> Path | None:
    """Line chart: cumulative article count over time (by file mtime)."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone

    data_points: list[tuple[float, str]] = []
    for md_file in WIKI_DIR.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        data_points.append((md_file.stat().st_mtime, md_file.parent.name))

    if not data_points:
        return None

    data_points.sort(key=lambda x: x[0])
    dates = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in data_points]
    cumulative = list(range(1, len(data_points) + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, cumulative, color="steelblue", linewidth=2)
    ax.fill_between(dates, cumulative, alpha=0.2, color="steelblue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Articles")
    ax.set_title("Knowledge Base Growth Over Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        path = _get_charts_dir() / f"knowledge-growth-{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[charts] Saved: {path}")
        return path
    plt.show()
    return None


def plot_backlink_graph(max_nodes: int = 50, save: bool = True) -> Path | None:
    """Network graph of wiki article backlinks."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("[charts] networkx not installed; skipping backlink graph.")
        return None

    from lore.compile.linker import build_backlink_map

    backlinks = build_backlink_map()
    if not backlinks:
        return None

    # Build graph from most-linked articles
    top_targets = sorted(backlinks.items(), key=lambda x: len(x[1]), reverse=True)[:max_nodes]

    G = nx.DiGraph()
    for target, sources in top_targets:
        G.add_node(target)
        for source in sources[:10]:
            G.add_edge(source, target)

    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2)
    node_sizes = [300 + 100 * G.in_degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="steelblue", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(f"Wiki Backlink Graph (top {G.number_of_nodes()} nodes)")
    ax.axis("off")
    plt.tight_layout()

    if save:
        path = _get_charts_dir() / f"backlink-graph-{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[charts] Saved: {path}")
        return path
    plt.show()
    return None


def plot_reward_history(save: bool = True) -> Path | None:
    """Plot RL reward history from trajectories DB."""
    from lore.config import TRAJECTORIES_DB
    if not TRAJECTORIES_DB.exists():
        return None

    try:
        import matplotlib.pyplot as plt
        from sqlitedict import SqliteDict
        import numpy as np

        rewards = []
        timestamps = []
        with SqliteDict(str(TRAJECTORIES_DB)) as db:
            for val in db.values():
                if isinstance(val, dict) and "reward" in val:
                    rewards.append(val["reward"])
                    timestamps.append(val.get("timestamp", 0))

        if len(rewards) < 2:
            return None

        # Sort by timestamp
        pairs = sorted(zip(timestamps, rewards))
        rewards = [r for _, r in pairs]

        # Rolling mean (window=10)
        window = min(10, len(rewards))
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(range(len(rewards)), rewards, alpha=0.4, s=20, color="steelblue", label="Per-query")
        ax.plot(range(window - 1, len(rewards)), rolling, color="red", linewidth=2, label=f"Rolling mean ({window})")
        ax.set_xlabel("Query #")
        ax.set_ylabel("Reward")
        ax.set_title("RL Reward History (Evolving Agent)")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()

        if save:
            path = _get_charts_dir() / f"reward-history-{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[charts] Saved: {path}")
            return path
        plt.show()
    except Exception as e:
        print(f"[charts] Reward history error: {e}")
    return None
