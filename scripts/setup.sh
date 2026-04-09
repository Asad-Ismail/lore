#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Checking for uv ==="
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Creating virtual environment and installing dependencies ==="
uv sync

echo "=== Verifying GPU ==="
uv run python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU-only (MPS: {torch.backends.mps.is_available()})')"

echo "=== Checking HF cache for required models ==="
uv run python - <<'EOF'
import os

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
# Also check SageMaker path for EC2 setups
ALT_CACHE = "/home/ec2-user/SageMaker/hf_cache/hub"
cache_dir = ALT_CACHE if os.path.exists(ALT_CACHE) else HF_CACHE

needed = ["Qwen3-4B", "Qwen3-1.7B"]
found = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
for model in needed:
    matches = [d for d in found if model.lower() in d.lower()]
    status = "FOUND: " + str(matches) if matches else "MISSING"
    print(f"  {model}: {status}")
EOF

echo "=== Setup complete ==="
