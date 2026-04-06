---
title: Model Quantization
category: concepts
created: 2026-04-06T09:27:50.530423+00:00
updated: 2026-04-06T09:27:50.530423+00:00
absorbed: true
---

## Model Quantization

Model quantization is the process of reducing the bit-precision of deep learning models to decrease model size and accelerate inference. This technique is crucial for deploying large language models (LLMs) on devices with limited memory and computational resources. Quantization methods generally fall into two categories: quantization-aware training (QAT) and post-training quantization (PTQ). QAT uses backpropagation to update quantized weights, while PTQ is training-free and often used for LLMs due to its scalability.

## Context

Model quantization is a critical area of research in machine learning and quantization, particularly for large-scale models. The goal is to maintain model accuracy while reducing computational and memory demands. This is essential for deploying models on edge devices and in environments with constrained resources. Various quantization techniques have been developed, including W8A8 quantization, which quantizes both weights and activations to INT8, and low-bit weight-only quantization, which quantizes only weights to low-bit integers.

## Key Claims

- Quantization reduces model size and accelerates inference by lowering bit-precision.
- Quantization techniques are divided into QAT and PTQ, with PTQ being more suitable for large models like LLMs.
- W8A8 quantization and low-bit weight-only quantization are two common settings for LLM quantization.
- AWQ (Activation-aware Weight Quantization) focuses on protecting "important" weights to improve quantized performance.
- GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that has shown significant improvements in compression and accuracy.
- SmoothQuant is an accurate and efficient post-training quantization method that addresses activation outliers by migrating quantization difficulty from activations to weights.

## Connections

- **Quantization-aware training (QAT)**: Involves training models with quantization applied, allowing for more accurate quantization but requiring more computational resources.
- **Post-training quantization (PTQ)**: A training-free approach that quantizes models after training, making it suitable for large models but potentially leading to performance degradation.
- **W8A8 quantization**: Quantizes both weights and activations to INT8, providing a balance between model size reduction and performance preservation.
- **Low-bit weight-only quantization**: Quantizes only weights to low-bit integers, reducing memory usage and improving inference speed.
- **AWQ**: Focuses on protecting important weights to maintain model accuracy during quantization.
- **GPTQ**: A post-training quantization method that uses a combination of techniques to achieve high compression and accuracy.
- **SmoothQuant**: An efficient post-training quantization method that addresses activation outliers by migrating quantization difficulty from activations to weights.

## Sources

- [awq-2306.00978]: Discusses the importance of quantization in reducing model size and accelerating inference, and introduces AWQ as a method that protects important weights.
- [gptq-2210.17323]: Details GPTQ, a post-training quantization method that achieves significant compression and accuracy, and discusses its performance on large models.
- [SmoothQuant: Accurate and Efficient]: Introduces SmoothQuant, an accurate and efficient post-training quantization method that addresses activation outliers by migrating quantization difficulty from activations to weights.