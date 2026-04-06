---
title: INT4
category: techniques
created: 2026-04-06T09:21:42.004489+00:00
updated: 2026-04-06T09:21:42.004489+00:00
absorbed: true
---

# INT4

## Definition

INT4 is a low-bit quantization format used in large language model (LLM) compression and acceleration, specifically for weight-only quantization. It reduces the memory footprint and computational cost of LLMs by representing weights as 4-bit integers, while maintaining high accuracy through activation-aware quantization techniques.

## Context

INT4 is a key component of the Activation-Aware Weight Quantization (AWQ) method, which aims to compress and accelerate LLMs for on-device deployment. AWQ leverages the distribution of activations to identify and protect salient weight channels, allowing for significant reductions in quantization error. This approach is particularly effective for models like LLaMA and Llama-2, where it achieves state-of-the-art performance on various benchmarks, including coding and math tasks.

## Key Claims

- **Activation-Aware Quantization**: AWQ identifies salient weight channels based on activation distribution, not weight distribution, to minimize quantization error.
- **Hardware Efficiency**: AWQ avoids mixed-precision quantization by scaling salient channels, leading to better performance on hardware with limited resources.
- **Generalization**: AWQ does not rely on backpropagation or reconstruction, allowing it to generalize across different domains and modalities without overfitting to the calibration set.
- **Performance**: AWQ outperforms existing methods on multiple benchmarks, including instruction-tuned LLMs and multi-modal models, achieving significant speedups on both desktop and mobile GPUs.
- **Deployment**: AWQ is implemented in TinyChat, an efficient inference framework that supports 4-bit quantization, achieving over 3× speedup compared to FP16 implementations.

## Connections

- **AWQ**: INT4 is the primary quantization format used in AWQ, which is designed to compress and accelerate LLMs by protecting salient weight channels.
- **TinyChat**: The inference framework developed alongside AWQ, which supports 4-bit quantization and achieves significant speedups on various hardware platforms.
- **GPTQ**: While GPTQ is a related quantization method, AWQ outperforms it in terms of accuracy and generalization, particularly for multi-modal models.
- **Post-Training Quantization (PTQ)**: INT4 is part of a broader category of PTQ methods, which aim to reduce model size and computational cost without retraining.
- **Low-Bit Quantization**: INT4 is a specific form of low-bit quantization, which is widely used in LLM compression and acceleration.

## Sources

- [awq-2306.00978, chunk 0]: Describes the AWQ method and its application to LLM compression and acceleration.
- [awq-2306.00978, chunk 1]: Discusses the challenges of LLM deployment and the benefits of low-bit quantization.
- [awq-2306.00978, chunk 2]: Explains the theoretical background of quantization and the role of activation-aware techniques.
- [awq-2306.00978, chunk 3]: Details the implementation of AWQ and the performance improvements achieved.
- [awq-2306.00978, chunk 4]: Provides experimental results showing the effectiveness of AWQ on various benchmarks.
- [awq-2306.00978, chunk 5]: Describes the optimization process for AWQ and the results on different model sizes.
- [awq-2306.00978, chunk 6]: Discusses the performance improvements achieved by AWQ on different hardware platforms.
- [awq-2306.00978, chunk 7]: Explains the technical details of AWQ, including the use of per-channel scaling and kernel fusion.
- [awq-2306.00978, chunk 8]: Presents the experimental results and comparisons with other quantization methods.
- [awq-2306.00978, chunk 9]: Provides further details on the performance of AWQ on different model architectures and tasks.
- [awq-2306.00978, chunk 10]: Discusses the application of AWQ to multi-modal models and the results on visual language tasks.
- [gptq-2210.17323, chunk 1]: Compares GPTQ with other quantization methods and highlights the limitations of current approaches.