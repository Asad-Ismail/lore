---
title: Quantization
category: concepts
created: 2026-04-06T09:24:13.397630+00:00
updated: 2026-04-06T09:24:13.397630+00:00
absorbed: true
---

**Quantization**

Quantization is a technique used in machine learning (ML) to reduce the precision of numerical values in a model, thereby decreasing the model's memory footprint and computational requirements. It is particularly useful for deploying large language models (LLMs) on edge devices where memory and processing power are limited. Quantization involves converting floating-point numbers into lower-bit integers, which can significantly reduce the size of the model while maintaining acceptable performance.

## Context

Quantization is a critical component in the deployment of LLMs on edge devices, where the high memory and computational demands of these models pose significant challenges. By reducing the bit precision of the model, quantization enables more efficient inference and faster execution on hardware with limited resources. This is especially important for applications such as virtual assistants, chatbots, and autonomous vehicles, where real-time performance and data privacy are paramount.

## Key Claims

- **Activation-aware Weight Quantization (AWQ)** is a hardware-friendly approach for LLM low-bit weight-only quantization. AWQ identifies salient weight channels based on activation distribution rather than weight distribution, which allows for significant reductions in quantization error.
- AWQ employs an equivalent transformation to scale salient weight channels, which is determined by collecting activation statistics offline. This method avoids the hardware-inefficient mix-precision quantization and generalizes well across different domains and modalities.
- AWQ outperforms existing work on various language modeling and domain-specific benchmarks, including coding and math tasks. It also achieves excellent quantization performance for instruction-tuned LMs and, for the first time, multi-modal LMs.
- The inference framework TinyChat, tailored for 4-bit on-device LLMs and VLMs, offers more than 3× speedup over the Huggingface FP16 implementation on both desktop and mobile GPUs. TinyChat also facilitates the deployment of the 70B Llama-2 model on mobile GPUs.

## Connections

- **Quantization-aware training (QAT)** and **post-training quantization (PTQ)** are two main approaches to quantization. QAT relies on backpropagation to update quantized weights, while PTQ does not. AWQ avoids the hardware-inefficient mix-precision implementation and does not rely on backpropagation or reconstruction.
- AWQ is related to **GPTQ**, another quantization method, but differs in its approach to identifying salient weight channels and its use of activation-aware scaling.
- The **TinyChat** framework is an efficient and flexible inference system that supports 4-bit quantized LLMs and VLMs, offering significant performance improvements over FP16 implementations.

## Sources

- [awq-2306.00978, chunk 0]: Introduction and overview of AWQ and TinyChat.
- [awq-2306.00978, chunk 1]: Discussion of the challenges of deploying LLMs on edge devices and the benefits of quantization.
- [awq-2306.00978, chunk 2]: Comparison of different quantization methods and the advantages of AWQ.
- [awq-2306.00978, chunk 3]: Detailed explanation of the activation-aware scaling method used in AWQ.
- [awq-2306.00978, chunk 4]: Analysis of the quantization error and the effectiveness of the scaling method.
- [awq-2306.00978, chunk 5]: Description of the optimization objective and the search space for the optimal scaling factor.
- [awq-2306.00978, chunk 6]: Discussion of the deployment of AWQ on edge platforms and the performance improvements achieved.
- [awq-2306.00978, chunk 7]: Results of experiments on various models and benchmarks, demonstrating the effectiveness of AWQ.
- [awq-2306.00978, chunk 8]: Additional results and comparisons with other quantization methods, highlighting the advantages of AWQ.