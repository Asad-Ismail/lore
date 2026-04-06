---
title: Memory Footprint
category: concepts
created: 2026-04-06T09:30:48.557705+00:00
updated: 2026-04-06T09:30:48.557705+00:00
absorbed: true
---

# Memory Footprint

## Definition
The memory footprint of a machine learning (ML) model refers to the amount of memory required to store and process the model's parameters and activations during inference. Reducing the memory footprint is crucial for deploying large language models (LLMs) on devices with limited memory, such as edge devices and mobile platforms.

## Context
The memory footprint of LLMs is a significant challenge due to their large parameter counts. For example, GPT-3 has 175B parameters, requiring 350GB in FP16, which exceeds the memory capacity of many GPUs. This has led to the development of various quantization techniques to reduce memory usage while maintaining model performance.

## Key Claims
- **AWQ (Activation-aware Weight Quantization)** is a hardware-friendly low-bit weight-only quantization method that reduces the memory footprint of LLMs by focusing on salient weight channels.
- **AWQ** outperforms existing quantization methods like GPTQ in terms of generalization and accuracy, achieving better performance on a variety of tasks and model sizes.
- **TinyChat** is an efficient inference framework that translates the theoretical memory savings from 4-bit LLMs into measurable speedups, achieving 3.2-3.3× average speedup on desktop, laptop, and mobile GPUs.
- **1-bit LLMs** like BitNet b1.58 offer a Pareto solution to reduce inference cost while maintaining model performance, with significant improvements in latency, memory, throughput, and energy consumption.
- **Mixed-precision decomposition** is a technique used to reduce the memory footprint of large transformers by representing a small number of large magnitude feature dimensions in 16-bit precision while the rest in 8-bit precision.

## Connections
- **AWQ** is closely related to **GPTQ**, but it addresses the issue of overfitting to the calibration set by using activation-aware quantization.
- **1-bit LLMs** like BitNet b1.58 are part of a broader trend towards lower-bit quantization, which includes methods like **INT4**, **INT8**, and **FP8**.
- **Mixed-precision decomposition** is a key technique in **LLM.int8()**, which is used to reduce the memory footprint of large transformers while maintaining performance.
- **QuIP#** is a new weight-only PTQ method that achieves state-of-the-art results in extreme compression regimes using three novel techniques: incoherence processing, lattice codebooks, and fine-tuning.

## Sources
- [awq-2306.00978, chunk 1]: Discusses the memory footprint of GPT-3 and the challenges of deploying large models on edge devices.
- [awq-2306.00978, chunk 5]: Details the activation-aware weight quantization method (AWQ) and its advantages over existing quantization techniques.
- [awq-2306.00978, chunk 6]: Explains the memory-bound nature of LLM inference and how AWQ reduces the weight memory by four times.
- [The Era of 1-bit LLMs:]: Introduces BitNet b1.58, a 1.58-bit LLM that achieves significant improvements in latency, memory, throughput, and energy consumption.
- [llm-int8-2208.07339, chunk 3]: Describes mixed-precision decomposition as a technique to reduce the memory footprint of large transformers.
- [QuIP#: Even Better LLM Quantization with]: Introduces QuIP#, a new weight-only PTQ method that achieves state-of-the-art results in extreme compression regimes.