#!/bin/bash
set -e
cd "$(dirname "$0")/.."

WITH_DEMO=0
RESET_DEMO=0

for arg in "$@"; do
    case "$arg" in
        --demo)
            WITH_DEMO=1
            ;;
        --reset-demo)
            WITH_DEMO=1
            RESET_DEMO=1
            ;;
        *)
            echo "Usage: bash scripts/setup.sh [--demo] [--reset-demo]"
            exit 1
            ;;
    esac
done

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

if [ "$WITH_DEMO" -eq 1 ]; then
    echo "=== Seeding demo workspace ==="
    if [ "$RESET_DEMO" -eq 1 ]; then
        uv run lore demo --reset
    else
        uv run lore demo
    fi
fi

echo "=== Setup complete ==="
if [ "$WITH_DEMO" -eq 1 ]; then
    echo "Try:"
    echo "  uv run lore status"
    echo "  uv run lore search \"active memory\""
    echo "  uv run lore-train suggest"
else
    echo "Open in your agent (Cursor, Claude Code) or run: uv run lore status"
fi
