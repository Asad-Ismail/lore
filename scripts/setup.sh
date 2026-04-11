#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Checking for uv ==="
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Installing dependencies ==="
uv sync

echo "=== Verifying device ==="
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'Device: CUDA ({torch.cuda.get_device_properties(0).name}, {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)')
elif torch.backends.mps.is_available():
    print('Device: MPS (Apple Silicon)')
else:
    print('Device: CPU only')
"

echo "=== Creating directories ==="
mkdir -p raw/{papers,articles,repos,images,notes}
mkdir -p wiki/{concepts,papers,meta}
mkdir -p outputs data

# Create seed wiki files if they don't exist
if [ ! -f wiki/_index.md ]; then
    cat > wiki/_index.md << 'EOF'
# Wiki Index

Agent-maintained catalog of all wiki articles. Read this first when answering a query.

## Concepts

## Papers

## Meta
EOF
    echo "  Created wiki/_index.md"
fi

if [ ! -f wiki/_log.md ]; then
    cat > wiki/_log.md << 'EOF'
# Wiki Log

Chronological record of operations. Append-only — newest entries at the bottom.

<!-- Format: ## [YYYY-MM-DD] operation | Title -->
<!-- Parseable with: grep "^## \[" _log.md | tail -5 -->
EOF
    echo "  Created wiki/_log.md"
fi

echo "=== Setup complete ==="
echo "Open in your agent (Cursor, Claude Code) or run: uv run lore status"
