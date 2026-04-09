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
EMBEDDINGS_DB = DATA_DIR / "embeddings.db"
TRAJECTORIES_DB = DATA_DIR / "trajectories.db"
LORA_CHECKPOINTS_DIR = DATA_DIR / "lora_checkpoints"
ABSORB_PENDING_FLAG = DATA_DIR / ".absorb_pending"

# ── HuggingFace cache ─────────────────────────────────────────────────────────
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/home/ec2-user/SageMaker/hf_cache"))
HF_HUB_DIR = HF_CACHE_DIR / "hub"

# ── Model IDs ─────────────────────────────────────────────────────────────────
# Primary inference model (wiki compilation, Q&A, health checks)
INFERENCE_MODEL_ID = "Qwen/Qwen3-4B"

# LoRA training target (smaller model that gets wiki knowledge baked in)
LORA_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

# Image captioning (for raw/images/)
IMAGE_MODEL_ID = "microsoft/Florence-2-base"

# Fallback instruction model
FALLBACK_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# ── Search ────────────────────────────────────────────────────────────────────
SEARCH_TOP_K = 8
SEARCH_RRF_K = 60          # RRF smoothing constant
SEARCH_RRF_ALPHA = 0.5     # Weight between TF-IDF and embedding

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

# ── Evolving agent / RL ───────────────────────────────────────────────────────
TRAIN_THRESHOLD = 10        # Trajectories before suggesting a retrain
GRPO_GROUP_SIZE = 4         # Responses generated per training prompt (G)
GRPO_BATCH_SIZE = 4         # Trajectory batch size
TRAIN_BUFFER_SAMPLE = 40    # Trajectories sampled per training run
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
OPD_SWITCH_STD = 0.05       # Switch from OPD to GRPO when reward std > this
OPD_BOOTSTRAP_N = 50        # Use OPD for first N trajectories

# ── Reward weights ────────────────────────────────────────────────────────────
REWARD_WEIGHT_GROUNDING = 0.40
REWARD_WEIGHT_CITATION = 0.25
REWARD_WEIGHT_COVERAGE = 0.20
REWARD_WEIGHT_FLUENCY = 0.15
GROUNDING_SIM_THRESHOLD = 0.25

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

# ── Compilation prompt ────────────────────────────────────────────────────────
COMPILE_SYSTEM_PROMPT = """\
You are compiling a personal ML/quantization research wiki. \
Given source excerpts about a concept, write a comprehensive wiki article.

Requirements:
- Lead with a clear 2-sentence definition/summary
- Use [[WikiLink]] syntax for any concept that deserves its own article
- Sections: ## Context, ## Key Claims, ## Connections, ## Sources
- In Key Claims, number each claim and note which source it comes from
- In Connections, explain how this relates to other wiki concepts using [[WikiLinks]]
- In Sources, list each source with specific section/table references
- 300–800 words, factual, no hallucination
- Write for a researcher who already knows ML fundamentals
"""

QUERY_SYSTEM_PROMPT = """\
You are answering questions about an ML research wiki. \
Answer based ONLY on the wiki context provided. \
If the answer is not fully covered, say so explicitly rather than speculating.
Cite the source articles you used using [[ArticleName]] syntax.
Be precise and technical.
"""
