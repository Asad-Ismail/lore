"""
Bootstrap script: first-time full compile from raw/.

Run this once after dropping a batch of papers into raw/ to build
the wiki from scratch. Equivalent to running lore-absorb but with
extra logging and a final status report.

Usage:
    python3 scripts/bootstrap_wiki.py
    python3 scripts/bootstrap_wiki.py --force   # recompile even existing articles
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from lore.ingest.pipeline import get_ingestion_stats
from lore.compile.compiler import absorb
from lore.index.store import rebuild_index
from lore.health.checker import run_health_check


def main():
    parser = argparse.ArgumentParser(description="Bootstrap the wiki from raw/")
    parser.add_argument("--force", action="store_true",
                        help="Recompile all articles even if up to date")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding index (faster, TF-IDF only)")
    args = parser.parse_args()

    print("=" * 60)
    print("Personal Wiki Bootstrap")
    print("=" * 60)

    # Step 1: Show ingestion state
    ingest_stats = get_ingestion_stats()
    print(f"\n[1/3] Ingestion state:")
    print(f"  Total chunks : {ingest_stats['total_chunks']}")
    print(f"  Unabsorbed   : {ingest_stats['unabsorbed']}")
    print(f"  Sources      : {ingest_stats['sources']}")

    if ingest_stats["total_chunks"] == 0:
        print("\nNo chunks found. Run 'lore-ingest <file>' first.")
        sys.exit(0)

    # Step 2: Absorb
    print(f"\n[2/3] Compiling wiki articles (force={args.force})...")
    stats = absorb(force=args.force)
    print(f"  New articles     : {stats['new_articles']}")
    print(f"  Updated articles : {stats['updated_articles']}")
    print(f"  Chunks processed : {stats['chunks_processed']}")
    print(f"  Concepts found   : {stats['concepts_found']}")

    # Step 3: Rebuild index
    print(f"\n[3/3] Building search index (embeddings={'off' if args.no_embeddings else 'on'})...")
    index_stats = rebuild_index(use_embeddings=not args.no_embeddings)
    print(f"  Articles indexed : {index_stats['articles']}")

    # Health check
    print("\n[health] Running initial health check...")
    health = run_health_check()
    h_stats = health.get("stats", {})
    print(f"  Articles  : {h_stats.get('total_articles', 0)}")
    print(f"  Words     : {h_stats.get('total_words', 0):,}")
    print(f"  Orphans   : {len(health.get('orphans', []))}")
    print(f"  Stubs     : {len(health.get('stubs', []))}")
    print(f"  Report    : wiki/_meta/health_report.md")

    print("\n" + "=" * 60)
    print("Bootstrap complete. Open wiki/ in Obsidian or run:")
    print("  lore query \"<your question>\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
