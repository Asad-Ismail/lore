#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-/tmp/lore-space-bundle}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/lore"

cp "$ROOT/space/app.py" "$OUT_DIR/app.py"
cp "$ROOT/space/README.md" "$OUT_DIR/README.md"
cp "$ROOT/space/requirements.txt" "$OUT_DIR/requirements.txt"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$ROOT/src/lore/" "$OUT_DIR/lore/"

echo "Built Lore Space bundle at $OUT_DIR"
