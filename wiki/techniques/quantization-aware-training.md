---
title: Quantization-aware Training
category: techniques
created: 2026-04-06T09:28:20.528739+00:00
updated: 2026-04-06T09:28:20.528739+00:00
absorbed: true
---

## Quantization-aware Training

Quantization-aware training (QAT) is a technique used in machine learning to train models with the knowledge that they will be deployed in a low-precision format. This approach allows for the preservation of model accuracy while significantly reducing computational and memory requirements. QAT involves modifying the training process to account for the effects of quantization, which is the process of reducing the precision of the model's parameters and activations.

## Context

Quantization-aware training is particularly relevant in the context of large language models (LLMs) and other deep learning models where memory and computational efficiency are critical. The increasing size of LLMs has led to higher memory and computational demands, making it challenging to deploy these models on edge devices and other resource-constrained hardware. QAT helps address these challenges by enabling the training of models that can be efficiently deployed in low-precision formats without significant loss of performance.

## Key Claims

- QAT involves training models with the knowledge that they will be deployed in a low-precision format, allowing for the preservation of model accuracy while reducing computational and memory requirements.
- QAT is more effective than post-training quantization (PTQ) in maintaining model accuracy under low-bit settings.
- AWQ (Activation-aware Weight Quantization) is a hardware-friendly low-bit weight-only quantization method that outperforms existing work in various tasks and model sizes.
- AWQ does not rely on backpropagation or reconstruction, preserving the generalization ability of LLMs on various domains and modalities without overfitting to the calibration set.
- TinyChat, an efficient inference framework, translates the theoretical memory savings from 4-bit LLMs to measured speedup, achieving consistent performance improvements across different hardware platforms.
- Bitnet.cpp, an efficient edge inference system for ternary LLMs, demonstrates significant speed advantages compared to baseline methods and provides lossless inference for BitNet b1.58.

## Connections

Quantization-aware training is closely related to other quantization techniques such as post-training quantization (PTQ) and low-bit weight quantization. While PTQ involves quantizing a model after training, QAT involves modifying the training process to account for the effects of quantization. AWQ and Bitnet.cpp are examples of methods that leverage QAT to achieve efficient and accurate inference on resource-constrained hardware. These methods are part of a broader effort to improve the efficiency and deployability of large language models.

## Sources

- awq-2306.00978: This paper introduces AWQ, an activation-aware weight quantization method for on-device LLM compression and acceleration. It details the methodology, implementation, and performance results of AWQ, including its application on various hardware platforms.
- Bitnet.cpp: Efficient Edge Inference for Ternary LLMs: This paper presents Bitnet.cpp, an efficient edge inference system for ternary LLMs. It discusses the methodology, performance results, and quality evaluation of Bitnet.cpp, including its application on different hardware platforms.
- Related Work: The paper also discusses related work in the field of LLM inference, including the application of LUT-based mpGEMM and other quantization techniques. It highlights the importance of efficient inference methods for deploying large language models on edge devices.