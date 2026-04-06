#!/bin/bash
# post_absorb.sh — Triggered after /lore absorb completes.
# Rebuilds the search index if absorb created new/updated articles.

REPO=/home/ec2-user/SageMaker/personal-knowledge
FLAG="$REPO/data/.absorb_pending"

if [ -f "$FLAG" ]; then
    echo "[hook] post_absorb: rebuilding search index..."
    cd "$REPO"
    python3 -m lore.cli.wiki_cli rebuild-index --no-embeddings 2>&1
    echo "[hook] Index rebuild complete."
    rm -f "$FLAG"
else
    echo "[hook] post_absorb: no pending absorb, skipping."
fi
