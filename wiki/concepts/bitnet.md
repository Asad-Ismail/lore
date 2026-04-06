---
title: BitNet
category: concepts
created: 2026-04-06T09:20:24.728568+00:00
updated: 2026-04-06T09:20:24.728568+00:00
absorbed: true
---

## Definition

BitNet is a 1-bit Large Language Model (LLM) variant that uses ternary weights (values of {-1, 0, 1}) to achieve high performance with significantly reduced latency, memory usage, and energy consumption compared to full-precision models like FP16 or BF16. BitNet b1.58, a specific implementation, introduces a 1.58-bit representation by adding a zero value to the original 1-bit BitNet, enabling efficient inference while maintaining performance.

## Context

BitNet emerged as a response to the growing demand for more efficient large language models (LLMs) that can operate on edge and mobile devices with limited computational resources. Traditional LLMs, while powerful, are often too resource-intensive for deployment in such environments. BitNet addresses this by leveraging ternary weights and a novel computation paradigm, enabling high-performance LLMs with significantly lower costs in terms of latency, memory, throughput, and energy consumption.

## Key Claims

- BitNet b1.58 matches the performance of full-precision LLMs (e.g., FP16) in terms of perplexity and end-task performance, while being significantly more cost-effective.
- BitNet b1.58 reduces the memory footprint and energy consumption of LLMs, making them more suitable for deployment on edge and mobile devices.
- BitNet b1.58 introduces a new scaling law and training recipe for LLMs, enabling high-performance and cost-effective models.
- BitNet b1.58 enables a new computation paradigm and opens the door for designing specific hardware optimized for 1-bit LLMs.
- BitNet b1.58 achieves lossless inference for ternary LLMs, demonstrating its efficiency and effectiveness in practical applications.

## Connections

- **1-bit LLMs**: BitNet is a key example of 1-bit LLMs, which use ternary weights to reduce computational and memory costs.
- **Quantization**: BitNet employs post-training quantization to reduce the precision of weights and activations, significantly reducing the memory and computational requirements of LLMs.
- **Edge and Mobile Devices**: BitNet is particularly well-suited for deployment on edge and mobile devices due to its low memory and energy consumption.
- **Hardware Optimization**: BitNet's new computation paradigm calls for the design of specific hardware optimized for 1-bit LLMs, which is an active area of research.
- **Performance Metrics**: BitNet b1.58 has been evaluated on various language tasks and benchmarks, demonstrating its effectiveness in maintaining performance while reducing costs.

## Sources

- [The Era of 1-bit LLMs:](https://aka.ms/GeneralAI) Abstract and detailed description of BitNet b1.58, including its performance metrics and comparisons with full-precision models.
- [The Era of 1-bit LLMs:](https://aka.ms/GeneralAI) Discussion of the computational and memory efficiency of BitNet b1.58, as well as its potential for hardware optimization.
- [The Era of 1-bit LLMs:](https://aka.ms/GeneralAI) Evaluation of BitNet b1.58 on various language tasks and benchmarks, including perplexity and end-task performance.
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://github.com/microsoft/BitNet/tree/paper) Description of the Bitnet.cpp implementation, which enables efficient edge inference for ternary LLMs.
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://github.com/microsoft/BitNet/tree/paper) Discussion of the design and implementation of the Bitnet.cpp library, including its performance improvements over existing methods.