---
title: 4-bit AWQ
category: concepts
created: 2026-04-06T09:32:00.169550+00:00
updated: 2026-04-06T09:32:00.169550+00:00
absorbed: true
---

## 4-bit AWQ

4-bit AWQ (Activation-aware Weight Quantization) is a technique for compressing and accelerating large language models (LLMs) by quantizing model weights to 4 bits while preserving activation precision. This approach enables efficient on-device inference by reducing memory bandwidth requirements and improving arithmetic intensity, leading to significant speedups on both GPUs and CPUs.

## Context

4-bit AWQ is part of a broader effort to optimize LLMs for deployment on edge devices and mobile platforms. By reducing the precision of model weights, AWQ minimizes memory usage and computational demands, making it feasible to run large models on hardware with limited resources. The technique is particularly effective for generation tasks, where the memory-bound nature of the workload becomes a bottleneck. The paper "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration" presents AWQ as a solution that achieves substantial speedups while maintaining model accuracy.

## Key Claims

- **Memory Bandwidth Bottleneck**: The generation stage of LLMs is memory-bound, with an arithmetic intensity of approximately 1 in FP16, which is well below the peak computational throughput of GPUs like the 4090. This limits the performance of on-device inference.
- **Arithmetic Intensity Improvement**: Quantizing weights to 4 bits increases the arithmetic intensity to 4 FLOPs/Byte, enabling the model to reach 4 TFLOPS of peak performance on GPUs.
- **Weight Access Dominance**: Weight access is the primary source of memory traffic in LLMs. Quantizing weights to 4 bits reduces the memory footprint by four times, leading to significant speedups.
- **Speedup Achievements**: TinyChat, a framework that implements 4-bit AWQ, achieves up to 3× speedup on LLMs like VILA-7B and VILA-13B on NVIDIA Jetson Orin. It also supports a wide range of models, including StarCoder, StableCode, Mistral, and Falcon, with consistent performance improvements over AutoGPTQ.
- **Robustness to Calibration Distributions**: AWQ demonstrates robustness to variations in calibration data distributions, with only minor increases in perplexity compared to GPTQ.

## Connections

4-bit AWQ is closely related to other quantization techniques such as GPTQ and llama.cpp, but it distinguishes itself through its activation-aware approach, which preserves activation precision while quantizing weights. It is also connected to the broader field of model compression, which includes techniques like pruning, quantization, and knowledge distillation. The use of SIMD-aware weight packing, as described in the paper, highlights the importance of hardware-aware optimizations in achieving efficient inference on modern processors.

## Sources

- awq-2306.00978: The paper "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration" presents the 4-bit AWQ technique, including its theoretical performance improvements and practical implementations in frameworks like TinyChat. The paper details the memory-bound nature of generation tasks and the effectiveness of 4-bit quantization in increasing arithmetic intensity.
- Table 10 in awq-2306.00978: This table shows the speedup achieved by TinyChat with 4-bit AWQ on various LLMs, including VILA-7B and VILA-13B, demonstrating the effectiveness of the technique across different models and hardware platforms.
- Figure 9 and Figure 10 in awq-2306.00978: These figures illustrate the speedup results from TinyChat on different GPU architectures, including RTX 4090, Jetson Orin, and Raspberry Pi, highlighting the versatility and performance benefits of 4-bit AWQ.