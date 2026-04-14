#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/deploy_space.sh <space-id> [bundle-dir]"
  exit 1
fi

SPACE_ID="$1"
BUNDLE_DIR="${2:-/tmp/lore-space-bundle}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "$ROOT/scripts/build_space_bundle.sh" "$BUNDLE_DIR"
hf repos create "$SPACE_ID" --type space --space-sdk gradio --public --exist-ok
hf upload "$SPACE_ID" "$BUNDLE_DIR" . --repo-type space --commit-message "Update Lore demo Space"

echo "Uploaded Lore demo Space bundle to $SPACE_ID"
