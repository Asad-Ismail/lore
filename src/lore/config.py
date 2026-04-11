"""Central configuration for the personal knowledge wiki system."""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = REPO_ROOT / "raw"
WIKI_DIR = REPO_ROOT / "wiki"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
HOOKS_DIR = REPO_ROOT / "hooks"

# ── Data files ────────────────────────────────────────────────────────────────
FINGERPRINTS_DB = DATA_DIR / "fingerprints.db"
TFIDF_INDEX_PATH = DATA_DIR / "tfidf_index.pkl"
QUESTION_TRACES_DB = DATA_DIR / "question_traces.db"
LORA_CHECKPOINTS_DIR = DATA_DIR / "lora_checkpoints"

# ── HuggingFace cache ─────────────────────────────────────────────────────────
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

# ── Model IDs ─────────────────────────────────────────────────────────────────
# Curiosity training target — learns your questioning patterns
LORA_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"


# ── Search ────────────────────────────────────────────────────────────────────
SEARCH_TOP_K = 8

# ── Wiki taxonomy ─────────────────────────────────────────────────────────────
WIKI_CATEGORIES = [
    "concepts",     # Core ML/quant ideas
    "techniques",   # Specific methods and algorithms
    "papers",       # Academic papers
    "models",       # Model architectures
    "datasets",     # Datasets and benchmarks
    "benchmarks",   # Evaluation benchmarks
    "people",       # Researchers
    "meta",         # Reading lists, open questions, agendas
]

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
GRPO_BATCH_SIZE = 4

# ── Curiosity training ───────────────────────────────────────────────────────
CURIOSITY_TRAIN_THRESHOLD = 3   # Question traces before suggesting training
CURIOSITY_BOOTSTRAP_N = 30       # SFT before switching to GRPO
CURIOSITY_GROUP_SIZE = 4         # Candidate questions per wiki state
CURIOSITY_REWARD_WEIGHT_GAP = 0.35
CURIOSITY_REWARD_WEIGHT_STYLE = 0.25
CURIOSITY_REWARD_WEIGHT_NOVELTY = 0.25
CURIOSITY_REWARD_WEIGHT_SPECIFICITY = 0.15

# ── Device helpers ───────────────────────────────────────────────────────────

def get_device() -> str:
    """Best available device: cuda → mps → cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_map():
    """
    device_map argument for from_pretrained().
    - CUDA:  "auto"  (let accelerate split across GPUs)
    - MPS:   {"": "mps"}  (single MPS device; accelerate doesn't enumerate MPS)
    - CPU:   {"": "cpu"}
    """
    import torch
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def get_torch_dtype():
    """
    Best dtype for the current device.
    - CUDA:  bfloat16  (full coverage on Ampere+)
    - MPS:   float16   (bfloat16 has incomplete op support on MPS as of PyTorch 2.3)
    - CPU:   float32
    """
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

