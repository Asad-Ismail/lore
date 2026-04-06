#!/bin/bash
set -e
cd /home/ec2-user/SageMaker/personal-knowledge

echo "=== Installing dependencies ==="
pip install peft trl sentence-transformers bitsandbytes rank-bm25 \
    pdfminer.six typer rich fastapi uvicorn sqlitedict networkx httpx

echo "=== Installing package ==="
pip install -e .

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else 'No GPU')"

echo "=== Checking HF cache for required models ==="
python3 - <<'EOF'
import os
HF_CACHE = "/home/ec2-user/SageMaker/hf_cache/hub"
needed = ["Qwen3-4B", "Qwen3-1.7B"]
found = os.listdir(HF_CACHE) if os.path.exists(HF_CACHE) else []
for model in needed:
    matches = [d for d in found if model.lower() in d.lower()]
    status = "FOUND: " + str(matches) if matches else "MISSING"
    print(f"  {model}: {status}")
EOF

echo "=== Setup complete ==="
