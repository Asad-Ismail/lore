---
title: INT3
category: concepts
created: 2026-04-06T09:22:19.830037+00:00
updated: 2026-04-06T09:22:19.830037+00:00
absorbed: true
---

## Definition

INT3 is a low-bit quantization format used in deep learning models to reduce the precision of weights from floating-point numbers to 3-bit integers. It is part of a broader set of quantization techniques aimed at compressing large language models (LLMs) while maintaining their performance.

## Context

INT3 is primarily used in the context of weight-only quantization for LLMs, where only the weights of the model are quantized to low-bit integers, while the activations remain in higher precision. This approach helps reduce the memory footprint and computational requirements of the model, making it more suitable for deployment on edge devices with limited resources. INT3 is often used in conjunction with activation-aware quantization techniques that prioritize preserving the importance of certain weights based on their activation patterns.

## Key Claims

- INT3 is a 3-bit integer quantization format used for weight-only quantization of LLMs.
- It is part of a family of quantization techniques that include W8A8 (8-bit weights and 8-bit activations) and W4A16 (4-bit weights and 16-bit activations).
- INT3 is used in the activation-aware weight quantization (AWQ) method, which aims to preserve the performance of LLMs by protecting important weights based on their activation distribution.
- INT3 is more hardware-efficient than higher-bit quantization formats like W16A16, allowing for faster inference on edge devices.
- INT3 has been shown to achieve better performance than other quantization methods like GPTQ (Gradient-based PTQ) in certain scenarios, particularly when using activation-aware scaling to protect important weights.

## Connections

- **AWQ (Activation-aware Weight Quantization)**: INT3 is a key component of the AWQ method, which uses activation-aware scaling to protect important weights and reduce quantization error.
- **GPTQ (Gradient-based Post-Training Quantization)**: INT3 is compared to GPTQ in several studies, with AWQ showing superior performance in some cases due to its activation-aware approach.
- **LLM Quantization**: INT3 is part of a broader set of quantization techniques used to reduce the size and computational cost of LLMs, enabling deployment on devices with limited resources.
- **Edge Computing**: INT3 is particularly relevant in the context of edge computing, where low-bit quantization is essential for reducing memory usage and improving inference speed on devices like CPUs and GPUs.

## Sources

- [awq-2306.00978, chunk 2]: Describes the context of model quantization and the different quantization techniques used for LLMs, including INT3.
- [awq-2306.00978, chunk 3]: Details the activation-aware weight quantization method (AWQ) and the use of INT3 to preserve important weights based on activation distribution.
- [awq-2306.00978, chunk 5]: Explains the mathematical formulation of the AWQ method, including the use of INT3 for weight quantization.
- [awq-2306.00978, chunk 7]: Discusses the performance of INT3 in the context of LLM quantization, comparing it to other formats like INT4 and W16A16.
- [awq-2306.00978, chunk 8]: Provides experimental results showing the effectiveness of INT3 in improving the performance of LLMs on different model sizes and architectures.
- [llm-int8-2208.07339, chunk 1]: Discusses the use of INT3 in the context of large language models, highlighting its role in enabling inference on very large models with consumer GPUs.