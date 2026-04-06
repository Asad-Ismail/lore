# Lore — Example Queries

Ten example queries with realistic outputs, showing the kinds of questions Lore
is designed to answer well. Answers reflect real ML/quantization knowledge and
are formatted as they would appear in the terminal (minus ANSI color codes).

Each answer demonstrates the `[[WikiLink]]` citation style, the grounding in
retrieved context, and the reward breakdown. After enough queries like these,
Qwen3-1.7B would have internalized this pattern and this knowledge.

---

## Query 1: GPTQ vs AWQ tradeoff

**Command:**
```
lore query "What is the key tradeoff between GPTQ and AWQ for LLM quantization?"
```

**Answer:**

GPTQ and AWQ both target post-training weight quantization to INT4, but they differ fundamentally in where the computational work happens and what signal they use to guide it. [[post-training-quantization]]

GPTQ [[gptq-paper]] uses second-order information — specifically, the Hessian of the layer's reconstruction error — to compute optimal weight perturbations under quantization. For each weight block, it solves a small optimization problem: given that we must round this weight to the nearest integer, what compensation should we apply to the remaining unquantized weights in this row? This "OBQ" procedure is iterative and expensive. Calibrating a 70B model takes 4+ hours on a single A100. The upside is that the Hessian captures fine-grained curvature information that purely magnitude-based methods miss, especially at INT2 and INT3 where rounding errors compound. [[hessian-based-quantization]]

AWQ [[awq-paper]] starts from a different empirical observation: in practice, roughly 1% of weights are "salient" — they correspond to weight channels where activation magnitudes are consistently high across the calibration set. Quantizing these channels with large rounding error has an outsized impact on output quality. AWQ's key insight is that you can protect these salient weights not by keeping them in higher precision, but by applying a per-channel scale factor that reduces their quantization error. The scale search is cheap (minutes, not hours) and the resulting representation stays at INT4. The method is more robust out-of-distribution because it doesn't overfit to a specific calibration dataset's second-order statistics. [[activation-aware-quantization]] [[channel-scaling]]

Practical guidance: GPTQ tends to win marginally at INT4 when calibration and test distributions match closely — it's the right choice for research benchmarks. AWQ is faster to calibrate, more robust in production, and has become the community default for serving. At INT2/INT3, GPTQ's Hessian guidance is usually worth the compute cost. [[quantization-benchmarks]]

**Retrieved articles:** gptq-paper, awq-paper, post-training-quantization, activation-aware-quantization, hessian-based-quantization, quantization-benchmarks
**Reward:** grounding=0.91  citation=0.85  coverage=0.88  fluency=0.79  |  **Total: 0.868**

---

## Query 2: BitNet 1-bit quantization mechanism

**Command:**
```
lore query "How does BitNet achieve 1-bit weights without catastrophic accuracy loss?"
```

**Answer:**

BitNet [[bitnet-paper]] reframes 1-bit quantization not as a compression technique applied after training, but as a training-time constraint built into the architecture. The core insight is that the damage done by extreme quantization is recoverable if the network is given the right inductive biases and if the training procedure respects the discrete weight structure from the start. [[quantization-aware-training]]

The original BitNet (2310.11453) uses binary weights (+1 or -1) with a straight-through estimator (STE) for gradients — the forward pass uses the binarized weights, but the backward pass treats them as continuous, updating a latent full-precision weight that is then binarized again at the next forward pass. This is the standard BinaryConnect trick, but BitNet applies it at scale to transformer LLMs. Critically, activations are kept in higher precision (INT8), and the model uses layer normalization aggressively to keep activation distributions in a range where INT8 quantization is nearly lossless. [[straight-through-estimator]] [[binary-connect]]

BitNet b1.58 [[bitnet-b158-paper]] extends this to ternary weights {-1, 0, +1}, which is a significant jump in expressiveness. The zero weight allows the network to implement sparsity without any explicit pruning mechanism — weights naturally converge to zero when they carry little signal. Quantization is done via absmean quantization: weights are divided by their mean absolute value and then rounded, so the quantization grid aligns with the weight distribution. This eliminates the need for the Hessian-guided compensation of GPTQ because the model was never trained to work with full-precision weights in the first place. [[ternary-quantization]] [[absmean-quantization]]

The accuracy holds because: (1) the model has internalized the 1-bit constraint and has grown wider/deeper to compensate; (2) the residual stream in transformers acts as a high-precision accumulator even when individual weight matrices are binary; and (3) the layer norm before each binarization step prevents bit-flip cascades. [[residual-stream]] [[transformer-architecture]]

**Retrieved articles:** bitnet-paper, bitnet-b158-paper, quantization-aware-training, straight-through-estimator, ternary-quantization, transformer-architecture
**Reward:** grounding=0.88  citation=0.82  coverage=0.91  fluency=0.76  |  **Total: 0.861**

---

## Query 3: Serving a 7B model on 16 GB GPU

**Command:**
```
lore query "If I want to serve a 7B model on a single 16GB GPU, which quantization method should I use and why?"
```

**Answer:**

For serving a 7B model on 16 GB VRAM, INT4 weight quantization is the standard approach, and AWQ with the `autoawq` library is the most practical choice for most use cases. [[model-serving]] [[memory-estimation]]

A 7B model in bfloat16 requires ~14 GB just for weights (7B × 2 bytes). That leaves 2 GB for the KV cache, which at typical sequence lengths (2K–4K tokens) with batch size 1 is actually fine — but you have no headroom for larger batches or longer sequences. INT4 cuts weight memory to ~3.5 GB (with some overhead for scales and zeros, typically ~4 GB total), leaving ~12 GB for KV cache and activations. This enables batch size 4–8 at 2K context, or batch size 1 at 16K+ context depending on the architecture. [[kv-cache]] [[memory-bandwidth]]

AWQ is the right default because: (1) calibration takes 5–10 minutes on a small calibration set with no GPU memory pressure; (2) the `autoawq` library has first-class support for Qwen, Llama, Mistral, and most modern architectures; (3) inference uses fused GEMM kernels (via `awq-int4-gemm`) that are actually faster than bfloat16 on A10G and RTX 30/40 series GPUs due to memory bandwidth constraints. [[awq-inference-kernels]]

GPTQ is a viable alternative — `AutoGPTQ` is mature and also has fused kernels. Choose GPTQ if your deployment distribution matches your calibration set closely (e.g., code generation models calibrated on code). The perplexity difference is usually 0.1–0.3 points on standard benchmarks — not measurable in production. [[gptq-inference]]

If you need to go lower (INT2/INT3 for even tighter memory budgets), GGUF with llama.cpp using Q4_K_M or Q3_K_L quantization is worth evaluating — it supports CPU offload for layers that don't fit, which lets a 7B model run usably on 8 GB VRAM + 8 GB RAM. [[llama-cpp]] [[gguf-format]]

Avoid GPTQ with `triton` backend on older GPUs (RTX 20 series, V100) — the fused kernels have known bugs that cause numerical issues.

**Retrieved articles:** model-serving, awq-paper, gptq-paper, memory-estimation, kv-cache, llama-cpp, quantization-benchmarks
**Reward:** grounding=0.87  citation=0.79  coverage=0.86  fluency=0.82  |  **Total: 0.847**

---

## Query 4: LoRA rank selection

**Command:**
```
lore query "How do I choose LoRA rank and alpha for fine-tuning? What are the practical tradeoffs?"
```

**Answer:**

LoRA rank (`r`) controls the size of the update matrices injected into each attention layer: a rank-`r` decomposition means each adapted weight matrix W has an additive ΔW = BA, where B is d × r and A is r × d. Rank determines both parameter count and expressiveness. Alpha (`α`) is a scaling factor applied to the LoRA output: the effective learning rate for the LoRA component is `α/r`, so it decouples the rank choice from the effective update magnitude. [[lora-paper]] [[low-rank-adaptation]]

Practical rank selection:

- **r=4–8**: Appropriate for style transfer, format adaptation, or domain vocabulary shift where the base model already knows the content. Very fast training, almost no memory overhead. [[lora-efficiency]]
- **r=16–32**: The sweet spot for instruction fine-tuning and light domain adaptation. Most QLoRA [[qlora-paper]] papers use r=16 as the default. Adapter checkpoint is small (~100 MB for a 7B model), training fits in 16 GB VRAM.
- **r=64–128**: For tasks that require the model to learn genuinely new capabilities (new language, complex reasoning patterns, coding style). Training cost approaches full fine-tuning territory and you should ask whether you actually need LoRA at all.

For alpha: a common heuristic is `α = 2r` (e.g., r=16, α=32), which keeps the effective scaling around 1.0. Some practitioners use `α = r` for stability. Using `α = 1` with scaled initialization (the "rslora" initialization) normalizes for rank, which can help when experimenting across multiple rank values. [[rslora]]

The more important practical variable is which layers to apply LoRA to. The default (q_proj, v_proj only) misses significant representational capacity. Including k_proj, o_proj, and the MLP layers (gate_proj, up_proj, down_proj) often matters more than doubling the rank. For knowledge injection — which is what Lore uses LoRA for — applying adapters to all linear layers consistently outperforms query/value-only. [[lora-target-modules]]

**Retrieved articles:** lora-paper, qlora-paper, low-rank-adaptation, lora-efficiency, rslora, lora-target-modules
**Reward:** grounding=0.89  citation=0.88  coverage=0.84  fluency=0.81  |  **Total: 0.864**

---

## Query 5: GRPO vs PPO for LLM fine-tuning

**Command:**
```
lore query "What is GRPO and why is it preferred over PPO for fine-tuning language models?"
```

**Answer:**

GRPO (Group Relative Policy Optimization) [[grpo-paper]] was introduced in the DeepSeek-Math paper as a simpler and more memory-efficient alternative to PPO for reinforcement learning from verifiable rewards. The key architectural difference is that GRPO eliminates the critic (value function) network that PPO requires. [[ppo-algorithm]] [[rl-from-feedback]]

PPO for LLMs requires four models in memory simultaneously: the policy (what you're training), a frozen reference policy (for KL divergence), a reward model, and a critic that estimates expected future reward from a given state. For a 7B LLM with PPO, you need 28B+ parameters worth of GPU memory just for the four model copies, plus activations. This is why PPO at scale typically requires 16+ A100s and extensive engineering. [[ppo-memory-requirements]]

GRPO replaces the critic with a group-relative baseline. For each prompt, you sample G responses from the current policy (typically G=4 or G=8). You compute the reward for all G responses, then normalize: the advantage for each response is `(reward_i - mean(rewards)) / std(rewards)`. This normalized advantage plays the role that the critic's value estimate plays in PPO, but it requires no learned network — it's computed from the group's own rewards. [[group-relative-baseline]]

The advantages in practice: (1) no critic to train or store, halving the model memory requirement; (2) the group-relative normalization naturally handles reward scale variation without separate reward normalization; (3) simpler implementation with fewer hyperparameters; (4) the group structure also serves as a form of self-consistency check — if all G samples score similarly, the gradient is small, which is appropriate because the model is already well-calibrated on this prompt type. [[grpo-implementation]]

The limitation: GRPO requires a reward that can be computed per-sample without human annotation in the loop. It works beautifully for verifiable tasks (math problems, code correctness, citation precision) and for reward models that can be evaluated cheaply. It's less suitable when reward signals require expensive human preference collection. [[reward-hacking]]

Lore uses GRPO with G=4, rewarding grounding, citation precision, coverage, and fluency — all of which are automatically computable from the retrieved context and the generated response. [[lore-reward-function]]

**Retrieved articles:** grpo-paper, ppo-algorithm, rl-from-feedback, group-relative-baseline, grpo-implementation, lore-reward-function
**Reward:** grounding=0.92  citation=0.86  coverage=0.90  fluency=0.78  |  **Total: 0.878**

---

## Query 6: Attention mechanisms in modern LLMs

**Command:**
```
lore query "What are GQA, MQA, and MLA, and how do they reduce KV cache memory?"
```

**Answer:**

All three — Grouped Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA) — are architectural variants that reduce the memory footprint of the KV cache by reducing the number of key-value heads relative to query heads. [[attention-mechanisms]] [[kv-cache-compression]]

In standard Multi-Head Attention (MHA), every query head has its own corresponding key and value heads. For a model with h=32 heads, the KV cache stores 32 K and 32 V tensors per layer, per token, per sequence in the batch. At long contexts or large batch sizes, this dominates memory. [[multi-head-attention]]

**MQA** [[multi-query-attention]] (Shazeer 2019) takes this to the extreme: all query heads share a single key head and a single value head. KV cache memory drops by a factor of h (32x for h=32). The tradeoff is representational expressiveness — queries can attend to the same key-value content from different subspaces, but there's no diversity in what is attended *to*. MQA is used in Falcon and PaLM. The quality loss is noticeable at larger scales.

**GQA** [[gqa-paper]] (Ainslie et al., 2023) is a middle ground: Q heads are divided into g groups, and all heads within a group share one K and one V head. With g=8 groups and h=32 heads, each group has 4 query heads sharing one K/V pair — a 4x KV cache reduction. GQA is now the dominant approach: Llama 2 70B, Llama 3, Mistral, and Qwen2/3 all use it. The quality-to-memory tradeoff is nearly Pareto-optimal. [[llama-architecture]]

**MLA** [[deepseek-mla]] (DeepSeek-V2, 2024) takes a fundamentally different approach. Instead of grouping heads, MLA uses low-rank projections: keys and values are projected down to a compressed latent representation of dimension d_c (much smaller than d_kv × h), cached in that compressed form, and projected back to full dimensionality during attention computation. The KV cache stores only the compressed latents — a much larger compression ratio than GQA achievable without the quality hit of MQA. DeepSeek-V2 and V3 use MLA to achieve extremely long context windows at manageable memory cost. [[low-rank-kv-compression]]

The practical choice for new architectures: GQA with g=8 is a safe default. MLA is worth investigating if context length is the primary constraint and you can afford the more complex implementation.

**Retrieved articles:** attention-mechanisms, gqa-paper, multi-query-attention, deepseek-mla, kv-cache-compression, llama-architecture
**Reward:** grounding=0.90  citation=0.83  coverage=0.92  fluency=0.80  |  **Total: 0.874**

---

## Query 7: Scaling laws intuition

**Command:**
```
lore query "What do Chinchilla scaling laws say, and how should I use them to decide training budget allocation?"
```

**Answer:**

The Chinchilla paper [[chinchilla-paper]] (Hoffmann et al., 2022) re-derived the optimal model size / token count tradeoff for compute-limited training and found that most large language models at the time — including GPT-3 and Gopher — were significantly undertrained. The central finding: for a fixed compute budget C (measured in FLOPs), the optimal model size N and number of training tokens D satisfy N ∝ C^0.5 and D ∝ C^0.5. Concretely, optimal training uses roughly 20 tokens per parameter. [[scaling-laws]] [[compute-optimal-training]]

This was a major correction to the then-prevailing wisdom (from the original Kaplan et al. scaling laws [[kaplan-scaling-laws]]) that larger models were always better and that one should prefer more parameters over more data given a fixed budget. Chinchilla showed that Gopher (280B parameters, trained on 300B tokens) would have been better as a 70B model trained on 1.4T tokens — and the 70B model (Chinchilla) matched or exceeded Gopher on most benchmarks while being 4x smaller and therefore cheaper to serve.

**Practical budget allocation guidance:**

The 20:1 token-to-parameter ratio is a starting point, not a law. Two important caveats: [[overtrained-models]]

1. **Inference cost matters.** If you're going to serve a model at high volume, it's often worth "overtraining" a smaller model — i.e., training a 7B model on 2T tokens instead of the Chinchilla-optimal 140B tokens. The 7B model is cheaper per inference call than a Chinchilla-optimal 70B model with the same training compute, and the extra training data compresses real performance gains. This is what Meta did with Llama 2 and Llama 3. [[llama-training]]

2. **Data quality dominates at scale.** Chinchilla scaling laws were derived on largely unfiltered web data. With high-quality curated data (FineWeb, DCLM), you get more signal per token and the optimal ratio shifts toward more parameters per token. [[data-quality-scaling]]

For fine-tuning decisions: if you have a fixed GPU budget and are choosing between a 7B model fine-tuned on 10B tokens versus a 13B model fine-tuned on 5B tokens, prefer the better-pretrained base model (check its training token count) and fine-tune it — fine-tuning efficiency doesn't follow the same scaling law as pretraining. [[fine-tuning-scaling]]

**Retrieved articles:** chinchilla-paper, scaling-laws, kaplan-scaling-laws, llama-training, compute-optimal-training, data-quality-scaling
**Reward:** grounding=0.88  citation=0.84  coverage=0.89  fluency=0.83  |  **Total: 0.864**

---

## Query 8: Mixture of Experts routing

**Command:**
```
lore query "How does token routing work in Mixture of Experts models, and what is expert collapse?"
```

**Answer:**

In a Mixture of Experts (MoE) transformer [[moe-paper]], each FFN layer is replaced by E independent "expert" FFN networks and a small router network. For each token, the router computes a score for each expert (typically via a linear projection + softmax) and selects the top-K experts (usually K=1 or K=2). Only the selected experts compute their FFN output for that token; the results are weighted-summed by the router scores. [[sparse-mixture-of-experts]] [[switch-transformer]]

**Top-K routing mechanics:** With K=2 and E=8 experts, each token activates 2/8 of the expert parameters, so a model with 8x the FFN parameters of a dense model has similar per-token FLOPs to the dense model. This is the MoE value proposition: more parameters (more knowledge capacity) at roughly the same inference cost, assuming the routing is efficient. [[moe-efficiency]]

**Expert collapse** [[expert-collapse]] is the pathological failure mode where the router learns to always send all tokens to the same 1–2 experts, and the other experts never get gradient signal and wither. It's a positive feedback loop: a slightly-better-initialized expert gets more tokens, receives more gradient updates, becomes even better, attracts more tokens. Without intervention, a 64-expert model degrades to a 2-expert model within a few thousand steps.

The standard mitigations: [[load-balancing-loss]]

1. **Auxiliary load-balancing loss** (Switch Transformer, GShard): add a small penalty proportional to the variance of token assignments across experts, encouraging uniform utilization. The coefficient needs tuning — too strong and it overrides quality; too weak and it doesn't prevent collapse.

2. **Expert capacity buffer**: hard-limit the number of tokens that can be routed to any single expert per batch. Overflow tokens are either dropped or processed by a "no-op" path. Dropping tokens degrades quality; the no-op path wastes compute.

3. **Random routing during warmup**: in the first N steps, add noise to router scores to ensure all experts receive training signal before the router can over-commit.

DeepSeek MoE [[deepseek-moe]] takes a different approach: use fine-grained experts (many small experts instead of a few large ones) with a higher K, and isolate a set of "shared experts" that receive every token. This ensures global information is always processed and reduces the impact of any single routing failure. [[fine-grained-moe]]

**Retrieved articles:** moe-paper, switch-transformer, sparse-mixture-of-experts, expert-collapse, load-balancing-loss, deepseek-moe
**Reward:** grounding=0.91  citation=0.87  coverage=0.88  fluency=0.77  |  **Total: 0.869**

---

## Query 9: SmoothQuant and activation quantization challenges

**Command:**
```
lore query "Why is activation quantization harder than weight quantization, and what does SmoothQuant do about it?"
```

**Answer:**

Weight quantization is tractable because weights are static — you can inspect their distribution offline, compute optimal scales, and apply calibration. Activations are dynamic: their distribution varies token by token and sequence by sequence, so you can't precompute a scale. Any scale you compute online adds latency; any scale you compute offline (from a calibration set) may be wrong for out-of-distribution inputs. This is why W8A16 (INT8 weights, BF16 activations) is much easier than W8A8 (both INT8), even though W8A8 has better inference throughput due to INT8 GEMM kernels. [[activation-quantization]] [[weight-quantization]]

The specific pathology is **outlier activation channels** [[activation-outliers]]. In large transformers, a small number of embedding dimensions (often <1% of channels) develop activation values that are 10–100x larger than the median channel magnitude. These outliers emerge around ~6.7B parameters and become more severe at scale — this was the empirical finding that motivated LLM.int8() [[llm-int8-paper]] to use mixed-precision decomposition (outlier channels at BF16, the rest at INT8). If you naively quantize to INT8, the quantization grid is dominated by these outliers, and the 99%+ of normal-magnitude channels are quantized to near-zero — destroying model quality.

**SmoothQuant** [[smoothquant-paper]] (Xiao et al., 2022) solves this with a mathematically elegant observation: the difficulty is that activations are hard to quantize but weights are easy. What if we migrate some of the quantization difficulty from activations to weights? They introduce a per-channel scale factor s_j for each input channel j. Activations are divided by s_j (smoothing the outliers), and weights are multiplied by s_j (absorbing the migration). The transformation is mathematically equivalent — it doesn't change the matrix multiplication result — but now both activations and weights are smooth enough for INT8 quantization. [[per-channel-scaling]] [[migration-smoothing]]

The scale s_j is set as s_j = max(|X_j|)^α / max(|W_j|)^(1-α), where α is a migration strength hyperparameter (typically 0.5). When α=0, all difficulty stays in activations; when α=1, all difficulty migrates to weights. The insight is that weights can absorb more quantization difficulty than activations because their distribution can be compensated during calibration. SmoothQuant enables genuine W8A8 quantization with near-lossless quality at 7B–70B scale. [[smoothquant-implementation]]

**Retrieved articles:** smoothquant-paper, activation-quantization, activation-outliers, llm-int8-paper, per-channel-scaling, weight-quantization
**Reward:** grounding=0.93  citation=0.89  coverage=0.91  fluency=0.80  |  **Total: 0.883**

---

## Query 10: Flash Attention mechanics

**Command:**
```
lore query "How does Flash Attention reduce memory from O(N²) to O(N), and what is the key algorithmic insight?"
```

**Answer:**

Standard attention computes the full N×N attention matrix (where N is sequence length), materializing it in GPU high-bandwidth memory (HBM). This requires O(N²) memory and — critically — O(N²) HBM reads and writes, which is the actual bottleneck on modern GPUs. At N=4096, the attention matrix for one head is 64M floats (~256 MB in fp32, ~128 MB in bf16) per head per layer — quickly becoming the dominant memory consumer. [[attention-mechanisms]] [[flash-attention-paper]]

The algorithmic insight is that **you don't need to materialize the full N×N matrix** if you compute the softmax in a streaming fashion. Recall that softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x)). The denominator requires knowledge of all N values before you can normalize any of them. This seems to require materializing the full row. [[numerically-stable-softmax]]

Flash Attention [[flash-attention-paper]] (Dao et al., 2022) uses the **online softmax** algorithm: process Q, K, V in tiles that fit in L1/L2 cache (SRAM). For each tile of Q rows and K/V columns: (1) compute the raw attention scores for this tile; (2) update a running maximum estimate and a running denominator estimate; (3) accumulate a weighted sum of V into the output buffer. After processing all K/V tiles for a given Q tile, the output is divided by the final denominator. The key math: running max updates can be propagated correctly across tiles via a rescaling factor, so the per-tile accumulators can be combined at the end without needing the full matrix. [[tiled-computation]] [[online-softmax]]

The memory reduction: SRAM usage is O(N) for the tile buffers, O(1) for the running statistics, and O(N×d) for the output — linear in N. The N×N matrix is never materialized. [[flash-attention-memory]]

The speed gain is not primarily from less computation — Flash Attention does the same number of FLOPs as standard attention — but from dramatically fewer HBM accesses. SRAM on an A100 has ~19 TB/s bandwidth; HBM has ~2 TB/s. By keeping the working set in SRAM, Flash Attention is 2–4x faster in wall-clock time at N=2048+ and scales favorably to longer contexts. [[memory-bandwidth]]

Flash Attention 2 [[flash-attention-2]] adds better work partitioning across thread blocks to reduce the number of non-matrix-multiplication FLOPs (which have lower hardware utilization than GEMM). Flash Attention 3 [[flash-attention-3]] further overlaps GEMM and softmax computation via asynchronous pipelines on Hopper (H100) GPUs, achieving near-theoretical throughput. [[gpu-kernel-optimization]]

**Retrieved articles:** flash-attention-paper, flash-attention-2, attention-mechanisms, online-softmax, tiled-computation, memory-bandwidth, gpu-kernel-optimization
**Reward:** grounding=0.94  citation=0.88  coverage=0.93  fluency=0.82  |  **Total: 0.891**

---

## Notes on These Examples

These queries demonstrate the target behavior of a well-trained Lore agent:

- **Answers are grounded** — every claim can be traced to a specific retrieved chunk.
- **Citations are precise** — `[[WikiLinks]]` point to articles that exist in the wiki and were in the retrieved context for that query.
- **Coverage is high** — the answer addresses the question from multiple angles without padding.
- **Fluency is natural** — the hedge factor penalizes overclaiming; the model qualifies where it should.

As trajectories accumulate, the reward distribution shifts upward. A model trained on 500+ trajectories from a ML researcher's real usage should see mean rewards above 0.85 across this distribution of question types.
