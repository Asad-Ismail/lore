# Lore — Architecture Deep Dive

This document is for contributors who want to understand the internals before
making changes. It covers the two-loop architecture, reward function math, the
GRPO training pipeline, the OPD bootstrap, divergence guard, and the hot-swap
inference server. Read it before touching anything in `src/lore/evolve/`.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Wiki Loop](#the-wiki-loop)
   - [Ingestion Pipeline](#ingestion-pipeline)
   - [Compilation Pipeline](#compilation-pipeline)
   - [Search Index](#search-index)
3. [The RL Loop](#the-rl-loop)
   - [Trajectory Capture](#trajectory-capture)
   - [Reward Function](#reward-function)
   - [OPD Bootstrap](#opd-bootstrap)
   - [GRPO Training](#grpo-training)
   - [Divergence Guard](#divergence-guard)
   - [Hot-Swap Mechanism](#hot-swap-mechanism)
4. [Data Schemas](#data-schemas)
5. [Configuration Reference](#configuration-reference)
6. [Failure Modes and Mitigations](#failure-modes-and-mitigations)
7. [Extension Points](#extension-points)

---

## System Overview

Lore runs two interleaved loops that share the `wiki/` directory as the
interface between them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LORE SYSTEM                                   │
│                                                                         │
│  raw/  ──► [ingest] ──► chunks.db                                       │
│                              │                                          │
│                         [absorb]  ◄── Qwen3-4B                          │
│                              │                                          │
│                           wiki/  ◄────────────────────────────────┐    │
│                              │                                     │    │
│                        [rebuild-index]                             │    │
│                              │                                     │    │
│                    tfidf_index.pkl + embeddings.db                 │    │
│                              │                                     │    │
│  lore query ──► [retrieval] ─┤                                     │    │
│                              │                                     │    │
│                    [Qwen3-1.7B + LoRA] ──► response                │    │
│                              │                                     │    │
│                    [reward function]                                │    │
│                              │                                     │    │
│                    [trajectory saved] ──► trajectories.db          │    │
│                              │                                     │    │
│                      [every 10 new]                                │    │
│                              │                                     │    │
│            ┌── n < 50? ──────┴──── n ≥ 50? ──────┐                │    │
│            │                                      │                │    │
│        [OPD: SFT on                          [GRPO: G=4            │    │
│         4B→1.7B distill]                    samples/prompt]        │    │
│            │                                      │                │    │
│            └──────────────────┬──────────────────┘                │    │
│                               │                                    │    │
│                        [save checkpoint]                           │    │
│                               │                                    │    │
│                       [divergence check]                           │    │
│                               │                                    │    │
│                   pass? ──────┴────── fail? ──► [rollback]        │    │
│                    │                                               │    │
│               [hot-swap] ──► inference server (port 8765) ────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

The wiki loop and RL loop are decoupled — you can run `lore absorb` without
ever running `lore query`, and `lore query` works without new absorb runs. The
feedback between them is indirect: as the wiki grows, retrieval quality
improves, which produces better grounding scores, which produces better GRPO
gradients, which produces a better inference model.

---

## The Wiki Loop

### Ingestion Pipeline

**Entry point:** `src/lore/ingest/pipeline.py` → `IngestPipeline.run(path_or_url)`

The pipeline is a linear chain of transforms:

```
input path/URL
    │
    ▼
[Parser]  ──── PDF → pdfminer text extraction
          ├─── Markdown → passthrough
          ├─── HTML/URL → httpx fetch + BeautifulSoup text extraction
          ├─── Image → Florence-2 caption
          └─── CSV → row concatenation
    │
    ▼
[Chunker]  ──── Recursive character splitter
           ├─── Chunk size: 800 tokens (tiktoken cl100k_base)
           ├─── Overlap: 100 tokens
           └─── Splits on: "\n\n", "\n", ". ", " "
    │
    ▼
[Fingerprinter]  ──── SHA-256 hash of raw text
                 └─── Check against fingerprints.db → skip if seen
    │
    ▼
[ChunkStore]  ──── Write to chunks.db (sqlitedict)
              └─── Key: "sha256_hex:chunk_index"
```

Deduplication is content-addressed: the same paper fetched twice (even from
different URLs) produces the same SHA-256 and is a no-op on the second ingest.

**Florence-2 image captioning:** When an image file is ingested, the pipeline
loads `microsoft/Florence-2-base` from the HF cache, runs
`<DETAILED_CAPTION>` inference, and treats the caption as the chunk text. The
image is copied to `raw/images/` and the caption is stored in `chunks.db` with
`source_type="image"`. The compiler uses this caption when writing wiki
articles that reference figures.

### Compilation Pipeline

**Entry point:** `src/lore/compile/compiler.py` → `WikiCompiler.absorb(force=False)`

Absorb works incrementally by default:

```
chunks.db
    │
    [select unprocessed chunks]  ←── "processed" flag per chunk
    │
    [group by source document]
    │
    [for each document group]:
        [prompt Qwen3-4B with all chunks]
            │
            system: You are a wiki compiler. Given source chunks, write
                    structured wiki articles in Obsidian markdown with
                    [[WikiLink]] citations. Use this taxonomy: concepts/,
                    techniques/, papers/, models/, datasets/, benchmarks/,
                    people/, meta/.
            │
            [parse response into articles]  ←── regex on ## headers
            │
            [snap WikiLinks]  ←── linker.snap_wikilinks(article_text)
            │
            [write to wiki/<category>/<slug>.md]
            │
            [mark chunks as processed]
```

The **WikiLink snapper** (`src/lore/compile/linker.py`) is a critical safety
net. Before any article is written to disk, every `[[Link]]` in the text is
checked against `wiki/_index.md` (the master article slug list). If the link
doesn't resolve, it tries:

1. Exact match (case-insensitive)
2. Levenshtein distance ≤ 2 (for typos and minor variations)
3. Substring match on slug tokens
4. Drop to plain text if no match found

This prevents the wiki from accumulating broken links even when the compiler
hallucinates plausible-sounding article names.

**Incremental compilation:** The `--force` flag reprocesses all chunks. Without
it, only chunks added since the last absorb run are sent to the compiler.
Articles are merged, not replaced: new content is appended under a `## Updates`
section with a timestamp, preserving article history.

### Search Index

**Entry point:** `src/lore/index/search.py`

The search index combines two retrieval mechanisms, fused with Reciprocal Rank
Fusion (RRF).

**TF-IDF index:**
- Scikit-learn `TfidfVectorizer` with `ngram_range=(1,2)`, `sublinear_tf=True`
- Fit on all wiki article bodies (concatenated title + content)
- Stored as sparse matrix in `data/tfidf_index.pkl`
- At query time: sparse dot product → top-K article slugs

**Embedding index:**
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
- One embedding per wiki article (mean-pooled over truncated body)
- Stored as numpy bytes in `data/embeddings.db` (sqlitedict)
- At query time: encode query → cosine similarity → top-K article slugs

**RRF fusion:**
```python
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

# For each candidate article slug:
combined_score = rrf_score(tfidf_rank) + rrf_score(embedding_rank)
```

RRF is robust to score distribution differences between the two systems —
you don't need to normalize TF-IDF scores against embedding cosine distances.
The fused ranking is used to select the top-12 article chunks passed to the
query model as context.

---

## The RL Loop

### Trajectory Capture

**Entry point:** `src/lore/evolve/trajectory.py` → `TrajectoryCapture.record()`

Every `lore query` call produces a trajectory record written to
`data/trajectories.db` (sqlitedict, key = `traj_{id:06d}`):

```python
@dataclass
class Trajectory:
    id: int
    timestamp: str               # ISO8601
    question: str                # raw user query
    retrieved_chunks: list[str]  # top-12 article bodies (text)
    retrieved_slugs: list[str]   # corresponding article slugs
    response: str                # model-generated answer
    reward: RewardBreakdown      # per-signal scores + total
    model_step: int              # LoRA checkpoint step at generation time
    mode: str                    # "base", "opd", or "grpo"
```

Trajectories are never deleted. They accumulate indefinitely and serve as
the training corpus, the reward history, and the audit log.

### Reward Function

**Entry point:** `src/lore/evolve/reward.py` → `compute_reward(trajectory)`

The reward is a weighted sum of four signals. Each signal is in [0, 1].

#### Signal 1: Grounding (weight = 0.40)

Measures whether the response stays close to the retrieved evidence.

```python
def grounding_score(response: str, chunks: list[str]) -> float:
    sentences = sent_tokenize(response)          # nltk
    grounded = 0
    for sent in sentences:
        scores = [
            cosine_tfidf(sent, chunk)            # TfidfVectorizer.transform
            for chunk in chunks
        ]
        if max(scores) >= GROUNDING_THRESHOLD:   # default 0.25
            grounded += 1
    return grounded / max(len(sentences), 1)
```

The threshold of 0.25 was tuned on 50 manually-rated trajectories to
approximately separate "stays close to evidence" from "goes off-script."
Increasing this threshold makes the reward more conservative (penalizes valid
synthesis across multiple sources); decreasing it is too permissive.

#### Signal 2: Citation (weight = 0.25)

Measures the precision of `[[WikiLink]]` citations.

```python
def citation_score(response: str, retrieved_slugs: list[str],
                   wiki_slugs: set[str]) -> float:
    cited = extract_wikilinks(response)          # regex \[\[([^\]]+)\]\]
    if not cited:
        return 0.5                               # neutral if no citations
    valid = [
        slug for slug in cited
        if slug in wiki_slugs                    # exists in wiki
        and slug in retrieved_slugs              # was in retrieved context
    ]
    return len(valid) / len(cited)               # precision
```

Note: the score is precision-only (not F1). We don't penalize for failing to
cite every retrieved article — that would reward citation-stuffing. We only
penalize for citing articles that weren't in the context (hallucinated links)
or don't exist in the wiki (broken links). The 0.5 neutral score for
no-citation responses prevents the model from learning to avoid citations
entirely to avoid precision penalties.

#### Signal 3: Coverage (weight = 0.20)

Measures completeness of the answer relative to the question. This signal
requires a judge model (Qwen3-4B) and is therefore more expensive to compute.

```python
COVERAGE_PROMPT = """
Rate how completely the following answer addresses the question.
Score 0-4:
  0 = off-topic or refuses to answer
  1 = superficial, misses key aspects
  2 = covers main points, misses some nuance
  3 = thorough, covers the question well
  4 = exceptional, addresses nuances and edge cases

Question: {question}
Answer: {response}

Respond with a single integer 0-4.
"""

def coverage_score(question: str, response: str) -> float:
    raw = judge_model.generate(COVERAGE_PROMPT.format(...))
    score = int(re.search(r"[0-4]", raw).group())
    return score / 4.0
```

Coverage is computed asynchronously and cached: the same (question, response)
pair always returns the same score. The judge model is loaded into a separate
process to avoid interfering with the inference model's GPU memory.

**Important:** The judge model (Qwen3-4B) must not be the same model being
trained. If the 1.7B model learns to generate responses that fool the 4B judge,
you get reward hacking rather than real improvement. The judge is frozen and
never updated by the training loop.

#### Signal 4: Fluency (weight = 0.15)

Measures response quality as a proxy for non-repetition and appropriate
epistemic confidence.

```python
def fluency_score(response: str) -> float:
    tokens = response.lower().split()
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0

    unique_bigram_ratio = len(set(bigrams)) / len(bigrams)

    # Hedge factor: penalize overconfident hedging-free responses
    hedge_words = {"likely", "typically", "generally", "usually",
                   "often", "may", "might", "can", "tends"}
    has_hedge = any(w in tokens for w in hedge_words)
    hedge_factor = 1.0 if has_hedge else 0.85

    return unique_bigram_ratio * hedge_factor
```

The unique bigram ratio penalizes repetitive responses (which often indicate
degenerate generation). The hedge factor provides a small incentive for
epistemically humble language — the model should not assert things that aren't
in the retrieved evidence with high confidence. The 0.85 factor was chosen to
be noticeable but not dominant.

#### Total Reward

```python
WEIGHTS = {
    "grounding": 0.40,
    "citation":  0.25,
    "coverage":  0.20,
    "fluency":   0.15,
}

total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
```

Total reward is in [0, 1]. In practice, a well-calibrated response after
training scores 0.80–0.90. Pathological responses (hallucinated citations,
ignoring retrieved context) score below 0.50.

### OPD Bootstrap

**Entry point:** `src/lore/evolve/trainer.py` → `TrainingManager.train_opd()`

**Why OPD?** Before the policy has enough data to self-improve via RL, GRPO
gradient estimates are high-variance. A model that generates uniformly bad
responses across G=4 samples sees near-zero advantage for all of them, producing
near-zero gradients — the model doesn't improve. The OPD bootstrap solves the
cold-start problem by initializing the small model's behavior on the research
domain before switching to RL.

**Implementation:**

For each trajectory in the buffer (trajectories 1–50):
1. Build the same prompt that the 1.7B model would have received (question +
   retrieved context)
2. Query Qwen3-4B with the same prompt — its response is the teacher response
3. SFT-train Qwen3-1.7B + LoRA to maximize the log-likelihood of the teacher
   response given the prompt

This is standard knowledge distillation, but online (using the actual inference
prompts) rather than offline (using a fixed dataset). The teacher response is
not cached and reused across training runs — a fresh teacher response is
generated for each new trajectory the first time it enters the training buffer.

```python
def train_opd(self, trajectories: list[Trajectory]):
    dataset = []
    for traj in trajectories:
        teacher_response = self.teacher_model.generate(
            build_prompt(traj.question, traj.retrieved_chunks)
        )
        dataset.append({
            "prompt": build_prompt(traj.question, traj.retrieved_chunks),
            "response": teacher_response,
        })

    self.sft_trainer.train(
        dataset,
        lora_config=self.lora_config,
        num_epochs=1,
        learning_rate=2e-4,
    )
```

The OPD phase ends after 50 trajectories, regardless of reward level. The
switch to GRPO is hard-coded, not adaptive. This is a deliberate simplicity
choice — an adaptive switch would require its own tuning.

### GRPO Training

**Entry point:** `src/lore/evolve/trainer.py` → `TrainingManager.train_grpo()`

GRPO generates G candidate responses per prompt, scores them all, normalizes
the rewards within the group, and updates the policy to favor higher-normalized-
reward responses. No critic network. No separate reward model (the reward
function is computed analytically).

```python
G = 4  # group size

def train_grpo(self, trajectories: list[Trajectory]):
    for traj in trajectories:
        prompt = build_prompt(traj.question, traj.retrieved_chunks)

        # Sample G responses from current policy
        responses = [
            self.policy_model.generate(prompt, do_sample=True, temperature=0.8)
            for _ in range(G)
        ]

        # Score each response
        rewards = [
            compute_reward(Trajectory(
                question=traj.question,
                retrieved_chunks=traj.retrieved_chunks,
                retrieved_slugs=traj.retrieved_slugs,
                response=r,
                ...
            )).total
            for r in responses
        ]

        # Normalize within group
        mean_r = statistics.mean(rewards)
        std_r  = statistics.stdev(rewards) + 1e-8   # epsilon for stability
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Policy gradient update
        for response, advantage in zip(responses, advantages):
            log_probs = self.policy_model.log_probs(prompt, response)
            loss = -advantage * log_probs.mean()    # REINFORCE objective
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
```

**KL regularization:** A KL penalty against the base model (no LoRA) is added
to prevent excessive policy drift:

```python
kl_penalty = KL_COEFF * kl_divergence(
    policy_logits, base_logits, prompt + response
)
total_loss = reinforce_loss + kl_penalty
```

`KL_COEFF = 0.05` by default. Increase if the model starts ignoring retrieved
context (over-relying on pre-training priors); decrease if training is too slow.

**LoRA configuration for training:**

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

All linear layers (not just Q/V) are adapted. This is important for knowledge
injection — the MLP layers carry much of the factual content, and limiting LoRA
to attention is insufficient for the research Q&A task.

**Training cadence:**

- Training is triggered after every 10 new trajectories
- Each training run processes all trajectories in the buffer (not just the last
  10) — the model is trained on the full accumulated history each time
- This means early trajectories have more training passes than late ones, which
  is a bias toward the model's initial research domain. If the research domain
  shifts significantly, consider `--reset-buffer` to start fresh

### Divergence Guard

**Entry point:** `src/lore/evolve/trainer.py` → `DivergenceGuard.check()`

After every training run and checkpoint swap, the divergence guard computes:

```python
def check(self) -> bool:
    """Returns True if the new checkpoint should be kept."""
    recent_rewards = [
        t.reward.total
        for t in self.trajectory_buffer.last_n(20)
    ]
    historical_rewards = [
        t.reward.total
        for t in self.trajectory_buffer.last_n(100)
    ]

    recent_mean = statistics.mean(recent_rewards)
    hist_mean   = statistics.mean(historical_rewards)
    hist_std    = statistics.stdev(historical_rewards) + 1e-8

    z_score = (recent_mean - hist_mean) / hist_std
    return z_score > -1.0    # keep if within 1 std of baseline
```

If the check returns `False`, the previous checkpoint is restored:

```python
if not divergence_guard.check():
    logger.warning(
        f"Divergence detected: z={z_score:.2f}. Rolling back to {prev_checkpoint}."
    )
    self.inference_server.load_checkpoint(prev_checkpoint)
    self.active_checkpoint = prev_checkpoint
```

The divergence guard uses the last 20 trajectories as the "recent" window
because GRPO can produce noisy reward estimates on small batches. 20 trajectories
is enough to detect a genuine performance regression while being recent enough
to reflect the new checkpoint's actual behavior.

**Known limitation:** The divergence guard fires *after* trajectories are already
collected with the new checkpoint. If the new checkpoint is significantly worse,
those 20 trajectories are lower-quality training data. A future improvement
would be to shadow-test new checkpoints on a held-out question set before
swapping them into production.

### Hot-Swap Mechanism

**Entry point:** `src/lore/evolve/server.py` → FastAPI app, port 8765

The inference server keeps the base model (Qwen3-1.7B) loaded in GPU memory
at all times and swaps LoRA adapters as training produces new checkpoints.

```python
# Simplified from server.py
class InferenceServer:
    def __init__(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_1_7B_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.active_adapter_path = None
        self._adapter_lock = threading.RLock()

    def load_checkpoint(self, checkpoint_path: str):
        """Thread-safe hot-swap of LoRA adapter."""
        with self._adapter_lock:
            if self.active_adapter_path is not None:
                self.base_model.disable_adapters()
                self.base_model.delete_adapter("active")

            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint_path,
                adapter_name="active",
            )
            self.active_adapter_path = checkpoint_path

    def generate(self, prompt: str, **kwargs) -> str:
        with self._adapter_lock:
            ...
```

The `RLock` ensures that a query in progress completes with the adapter it
started with before a swap occurs. Swaps are rare (every ~10 queries) so
lock contention is negligible.

**VRAM usage during hot-swap:**

During a swap, both the old and new adapter weights are briefly resident in
VRAM (the new adapter is loaded before the old one is deleted). For Qwen3-1.7B
with r=16, the LoRA adapter is ~100 MB — well within the safety margin.

---

## Data Schemas

### `chunks.db` (sqlitedict)

Key: `f"{sha256_hex}:{chunk_index}"`

```python
@dataclass
class Chunk:
    sha256: str           # hex of raw source text
    chunk_index: int      # position within source document
    source_path: str      # original file path or URL
    source_type: str      # "pdf" | "markdown" | "html" | "image" | "csv"
    text: str             # chunk text (800 tokens max)
    title: str            # extracted or inferred title
    processed: bool       # True after absorb has compiled this chunk
    ingested_at: str      # ISO8601 timestamp
```

### `trajectories.db` (sqlitedict)

Key: `f"traj_{id:06d}"`

```python
@dataclass
class Trajectory:
    id: int
    timestamp: str
    question: str
    retrieved_chunks: list[str]
    retrieved_slugs: list[str]
    response: str
    reward: RewardBreakdown    # grounding, citation, coverage, fluency, total
    model_step: int
    mode: str                  # "base" | "opd" | "grpo"
```

### `embeddings.db` (sqlitedict)

Key: article slug (e.g., `"techniques/gptq"`)

```python
@dataclass
class ArticleEmbedding:
    slug: str
    embedding: bytes          # np.float32 array serialized as bytes
    text_hash: str            # SHA-256 of article body (for staleness check)
    embedded_at: str          # ISO8601
```

---

## Configuration Reference

All configuration is in `src/lore/config.py`. Key settings:

```python
@dataclass
class LoreConfig:
    # Paths
    raw_dir:             Path = ROOT / "raw"
    wiki_dir:            Path = ROOT / "wiki"
    data_dir:            Path = ROOT / "data"
    hf_cache_dir:        Path = Path("/home/ec2-user/SageMaker/hf_cache")

    # Models
    compiler_model:      str = "Qwen/Qwen3-4B"
    agent_model:         str = "Qwen/Qwen3-1.7B"
    vision_model:        str = "microsoft/Florence-2-base"
    embed_model:         str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion
    chunk_size:          int = 800       # tokens
    chunk_overlap:       int = 100

    # Search
    top_k_retrieval:     int = 12
    grounding_threshold: float = 0.25

    # Reward
    reward_weights: dict = field(default_factory=lambda: {
        "grounding": 0.40,
        "citation":  0.25,
        "coverage":  0.20,
        "fluency":   0.15,
    })

    # Training
    train_every_n_trajectories: int = 10
    opd_phase_trajectories:     int = 50
    grpo_group_size:            int = 4
    grpo_temperature:           float = 0.8
    kl_coeff:                   float = 0.05
    learning_rate:              float = 2e-4

    # LoRA
    lora_r:              int = 16
    lora_alpha:          int = 32
    lora_dropout:        float = 0.05

    # Divergence guard
    divergence_recent_window:     int = 20
    divergence_historical_window: int = 100
    divergence_z_threshold:       float = -1.0
```

---

## Failure Modes and Mitigations

### WikiLink hallucination

**Symptom:** Articles contain `[[links]]` to articles that don't exist.
**Cause:** Qwen3-4B generates plausible-sounding article names for concepts
it knows about but that haven't been ingested yet.
**Mitigation:** `linker.snap_wikilinks()` runs on every article before it
reaches disk. Unresolvable links are degraded to plain text.
**If you see it anyway:** Run `lore cleanup` to audit and fix existing articles.

### Reward hacking via citation stuffing

**Symptom:** Model generates responses that consist mostly of `[[WikiLinks]]`
with minimal prose, achieving high citation precision by avoiding prose claims.
**Cause:** The citation signal rewards precision (not recall), and the grounding
signal requires sentences — a response with no sentences scores 0 on grounding.
**Mitigation:** The fluency signal penalizes low unique-bigram ratio. A
citation-stuffed response would have low fluency. Monitor the signal breakdown
in `lore-train status` — if citation is high and fluency is low, reduce
`reward_weights.citation` slightly.

### GRPO collapse (all rewards converge to same value)

**Symptom:** Training loss goes to zero but response quality doesn't improve.
`lore-train status` shows std(rewards) ≈ 0 for the group samples.
**Cause:** The policy has converged to a mode that scores consistently on all
four signals — it's found a local equilibrium. The group-relative advantage
is near zero, so gradients vanish.
**Mitigation:** Increase `grpo_temperature` from 0.8 to 1.0–1.2 to increase
diversity in the G samples. If that doesn't help, roll back several checkpoints
and retrain from a more exploratory state.

### OPD teacher-student gap

**Symptom:** After OPD, the 1.7B model sounds like a 4B model but scores
worse on actual reward metrics than the base 1.7B model did.
**Cause:** The 1.7B model doesn't have the capacity to reproduce the 4B
teacher's responses verbatim. It overfits to the style without capturing
the content, which actually hurts grounding scores.
**Mitigation:** Reduce `opd_phase_trajectories` to 25–30 and switch to GRPO
earlier. GRPO is self-referential (the 1.7B model competes against itself),
so the model doesn't need to match a larger model — it just needs to improve
relative to its own baseline.

### Embedding index staleness

**Symptom:** `lore search` returns outdated results after new absorb runs.
**Cause:** Embeddings are not recomputed automatically after absorb.
**Mitigation:** Run `lore rebuild-index` after any absorb run. The
`hooks/post_absorb.sh` hook does this automatically if Claude Code hooks
are configured.

---

## Extension Points

### Adding a new reward signal

1. Implement a function `def my_signal(trajectory: Trajectory) -> float` in
   `src/lore/evolve/reward.py`. Return a value in [0, 1].
2. Add your signal to `RewardBreakdown` dataclass.
3. Add a weight to `LoreConfig.reward_weights`. The weights must sum to 1.0.
4. Wire the signal into `compute_reward()`.
5. Write a test in `tests/test_reward.py`.

The most valuable addition would be a **factual consistency signal** that
checks whether claims in the response are entailed by the retrieved chunks
using an NLI model (e.g., `cross-encoder/nli-deberta-v3-small`). This would
catch responses that are fluent and cited but internally inconsistent with
the evidence.

### Adding a new ingest parser

1. Subclass `BaseParser` in `src/lore/ingest/parsers/base.py`.
2. Implement `can_handle(path_or_url: str) -> bool` and
   `parse(path_or_url: str) -> str`.
3. Register the parser in `src/lore/ingest/parsers/__init__.py`.

Parsers are tried in registration order; the first matching parser wins.

### Adding a new render target

1. Create `src/lore/render/my_format.py`.
2. Implement `render(topic: str, wiki_dir: Path, output_dir: Path) -> Path`.
3. Add a subcommand to `src/lore/cli/lore_cli.py` under `render`.

### Replacing the embedding model

The embedding model is used for both indexing and retrieval. To swap it:
1. Update `LoreConfig.embed_model`.
2. Run `lore rebuild-index --full` to recompute all embeddings.
3. Existing embeddings are automatically invalidated by the text hash check
   in `ArticleEmbedding`.

`BAAI/bge-small-en-v1.5` is a good upgrade from `all-MiniLM-L6-v2` if you
have the memory budget (slightly larger, significantly better retrieval).

### Training on a different base model

The RL loop is not tied to Qwen3-1.7B. To use a different target model:
1. Update `LoreConfig.agent_model`.
2. Verify the model supports PEFT LoRA (any HuggingFace CausalLM does).
3. Update `lora_target_modules` if the architecture uses different layer names.
4. Run `lore-train rollback --all` to discard existing checkpoints.
5. Retrain from scratch — checkpoints are model-specific.

---

*Last updated by the Lore system. For questions about this document, open an
issue or run `lore query "how does the GRPO training loop work in Lore"`.*
