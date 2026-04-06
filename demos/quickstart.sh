#!/usr/bin/env bash
# =============================================================================
#  Lore — Quickstart Demo
#  Run this script to go from zero to a trained personal research agent.
#
#  What this script does:
#    1. Installs dependencies
#    2. Ingests two foundational quantization papers from arXiv
#    3. Compiles them into a structured wiki
#    4. Runs three example queries (with real answers from the local model)
#    5. Shows the system status dashboard
#    6. Demonstrates how to trigger manual LoRA training
#
#  Requirements:
#    - Python 3.10+
#    - CUDA GPU recommended (A10G / RTX 3090+ / 24 GB VRAM ideal)
#    - ~15 GB free disk space for model cache
#    - HuggingFace models pre-downloaded to hf_cache/ (see setup.sh)
#
#  Usage:
#    bash demos/quickstart.sh
#    bash demos/quickstart.sh --fast   # skip training, shorter queries
# =============================================================================

set -euo pipefail

FAST_MODE=false
[[ "${1:-}" == "--fast" ]] && FAST_MODE=true

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${RESET}\n"; }
step()    { echo -e "${GREEN}▶${RESET} $*"; }
note()    { echo -e "${YELLOW}ℹ${RESET}  $*"; }
success() { echo -e "${GREEN}✓${RESET} $*"; }

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
header "Lore Quickstart"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: python3 not found. Install Python 3.10+.${RESET}"
    exit 1
fi

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "$PYTHON_MINOR" -lt 10 ]]; then
    echo -e "${RED}Error: Python 3.10+ required, found 3.${PYTHON_MINOR}.${RESET}"
    exit 1
fi

# Run from the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
note "Working directory: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
header "Step 1: Installing Dependencies"

step "Running setup.sh (installs Python packages, verifies GPU and models)..."
bash scripts/setup.sh

step "Installing Lore CLI entry points..."
pip install -e . --quiet

success "Installation complete."
echo ""
note "CLI commands available: lore, lore-train"

# ---------------------------------------------------------------------------
# 2. Ingest papers
# ---------------------------------------------------------------------------
header "Step 2: Ingesting Papers"

note "We'll ingest three foundational quantization papers:"
note "  • AWQ: Activation-aware Weight Quantization (2306.00978)"
note "  • GPTQ: Post-training quantization for GPTs (2210.17323)"
note "  • BitNet: Scaling 1-bit Transformers (2310.11453)"

step "Ingesting AWQ paper..."
lore ingest https://arxiv.org/abs/2306.00978
success "AWQ ingested."

step "Ingesting GPTQ paper..."
lore ingest https://arxiv.org/abs/2210.17323
success "GPTQ ingested."

step "Ingesting BitNet paper..."
lore ingest https://arxiv.org/abs/2310.11453
success "BitNet ingested."

echo ""
note "Papers are now chunked and stored in data/chunks.db."
note "Raw source files are preserved in raw/papers/ — never modified."

# ---------------------------------------------------------------------------
# 3. Compile into wiki articles
# ---------------------------------------------------------------------------
header "Step 3: Compiling Wiki"

step "Running lore absorb (Qwen3-4B reads chunks → writes wiki/*.md)..."
note "This takes 2–5 minutes depending on GPU. Grab a coffee."
echo ""

lore absorb

echo ""
success "Wiki compiled. Articles written to wiki/."
echo ""
note "New articles created (roughly):"
note "  wiki/techniques/awq.md"
note "  wiki/techniques/gptq.md"
note "  wiki/techniques/bitnet.md"
note "  wiki/concepts/post-training-quantization.md"
note "  wiki/concepts/weight-quantization.md"
note "  wiki/concepts/quantization-error.md"
note "  wiki/papers/awq-paper.md"
note "  wiki/papers/gptq-paper.md"
note "  wiki/papers/bitnet-paper.md"
note ""
note "Open wiki/ as an Obsidian vault to browse the graph view."

# ---------------------------------------------------------------------------
# 4. Rebuild search index
# ---------------------------------------------------------------------------
header "Step 4: Building Search Index"

step "Rebuilding TF-IDF + embedding indexes..."
lore rebuild-index
success "Search index ready."

# ---------------------------------------------------------------------------
# 5. Example queries
# ---------------------------------------------------------------------------
header "Step 5: Example Queries"

note "Running three queries. Each is captured as a training trajectory."
note "After 10 trajectories, training fires automatically in the background."
echo ""

# Query 1
echo -e "${BOLD}Query 1: The fundamental tradeoff${RESET}"
echo "─────────────────────────────────────────────────"
lore query "What is the key tradeoff between GPTQ and AWQ for LLM quantization?"
echo ""

# Give the model a moment between queries
sleep 2

# Query 2
echo -e "${BOLD}Query 2: BitNet's approach${RESET}"
echo "─────────────────────────────────────────────────"
lore query "How does BitNet achieve 1-bit weights without catastrophic accuracy loss?"
echo ""

sleep 2

# Query 3
echo -e "${BOLD}Query 3: Practical guidance${RESET}"
echo "─────────────────────────────────────────────────"
lore query "If I want to serve a 7B model on a single 16GB GPU, which quantization method should I use and why?"
echo ""

success "3 queries complete. Trajectories captured."
note "Run 'lore status' to see the training queue depth."

# ---------------------------------------------------------------------------
# 6. Status dashboard
# ---------------------------------------------------------------------------
header "Step 6: Status Dashboard"

lore status

# ---------------------------------------------------------------------------
# 7. Manual training trigger
# ---------------------------------------------------------------------------
header "Step 7: LoRA Training"

if $FAST_MODE; then
    note "Fast mode: skipping training. Run 'lore-train train' manually."
else
    note "The system auto-trains after every 10 trajectories."
    note "Since we only ran 3 queries, let's trigger training manually."
    echo ""

    step "Checking training status before..."
    lore-train status
    echo ""

    step "Triggering manual training run (OPD phase — teacher distillation)..."
    note "This will run for 1–3 minutes. Training in foreground for demo visibility."
    note "In normal use, this happens in a background subprocess."
    echo ""

    lore-train train

    echo ""
    success "Training complete."
    echo ""

    step "Checking training status after..."
    lore-train status
    echo ""

    note "The inference server (if running) would have hot-swapped the new adapter."
    note "Start the server with: lore-train serve"
fi

# ---------------------------------------------------------------------------
# 8. What's next
# ---------------------------------------------------------------------------
header "What's Next"

cat << 'EOF'
You now have a working Lore installation with:
  • 3 papers ingested and compiled into ~9 wiki articles
  • A search index built from those articles
  • 3 training trajectories captured
  • (Optional) A new LoRA checkpoint trained on those trajectories

Next steps:

  1. Feed more papers:
       lore ingest https://arxiv.org/abs/2208.07339   # LLM.int8()
       lore ingest https://arxiv.org/abs/2309.14717   # SqueezeLLM
       lore absorb

  2. Explore the wiki in Obsidian:
       Open Obsidian → File → Open Vault → select wiki/

  3. Run the health audit:
       lore health

  4. Generate a research report:
       lore render report "post-training quantization methods"

  5. Start the inference server for programmatic access:
       lore-train serve   # FastAPI at http://localhost:8765

  6. Ingest your own PDFs:
       lore ingest ~/Downloads/my-paper.pdf
       lore absorb

See demos/example_queries.md for 10 more example queries with expected outputs.
See demos/architecture.md for a deep dive into the RL loop implementation.

EOF

success "Quickstart complete. Happy researching."
