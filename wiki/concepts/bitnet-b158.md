---
title: BitNet b1.58
category: concepts
created: 2026-04-06T09:33:18.407640+00:00
updated: 2026-04-06T09:33:18.407640+00:00
absorbed: true
---

## Definition

BitNet b1.58 is a 1.58-bit Large Language Model (LLM) that uses ternary weights (values of {-1, 0, 1}) to achieve high performance while significantly reducing computational, memory, and energy costs. It represents a significant advancement in the field of low-bit LLMs, offering a new scaling law and computation paradigm that enables efficient inference on edge and mobile devices.

## Context

BitNet b1.58 is part of a broader movement toward low-bit LLMs, which aim to reduce the computational and energy demands of large language models without sacrificing performance. This movement is driven by the need for more efficient and scalable AI solutions, particularly for deployment on resource-constrained devices. BitNet b1.58 builds on the success of earlier 1-bit LLMs like BitNet, but introduces a key innovation by using ternary weights, which allows it to achieve a balance between performance and efficiency.

## Key Claims

- BitNet b1.58 achieves performance comparable to full-precision (FP16) LLMs while being significantly more efficient in terms of latency, memory, and energy consumption.
- It introduces a new computation paradigm that enables efficient inference on edge and mobile devices.
- BitNet b1.58 is a Pareto improvement over existing LLMs, offering better performance with lower costs.
- It enables a new scaling law for LLMs, allowing for the development of high-performance, cost-effective models.
- BitNet b1.58 is optimized for 1.58-bit weights and 8-bit activations, making it suitable for deployment on a wide range of hardware, including CPUs and GPUs.

## Connections

- **BitNet**: BitNet b1.58 is an extension of the BitNet architecture, which was introduced as a 1-bit LLM. BitNet b1.58 improves upon BitNet by introducing ternary weights, which allows for a more efficient and effective model.
- **Low-bit LLMs**: BitNet b1.58 is part of the broader trend toward low-bit LLMs, which aim to reduce the computational and energy demands of large language models. This trend is driven by the need for more efficient and scalable AI solutions, particularly for deployment on resource-constrained devices.
- **Edge and Mobile Devices**: BitNet b1.58 is optimized for deployment on edge and mobile devices, where computational and memory resources are limited. This makes it particularly useful for applications that require real-time inference on devices with limited processing power.
- **Hardware Optimization**: BitNet b1.58 enables the development of new hardware optimized for 1-bit LLMs, which could significantly improve the performance and efficiency of LLMs on specialized hardware.

## Sources

- [The Era of 1-bit LLMs:](https://aka.ms/GeneralAI) This paper introduces BitNet b1.58 as a 1.58-bit LLM that uses ternary weights to achieve high performance while significantly reducing computational, memory, and energy costs. The paper discusses the architecture of BitNet b1.58, its performance compared to full-precision LLMs, and its potential for deployment on edge and mobile devices.
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://github.com/microsoft/BitNet/tree/paper) This paper introduces Bitnet.cpp, an inference system optimized for BitNet b1.58 and ternary LLMs. The paper discusses the challenges of implementing efficient inference for ternary LLMs and presents Bitnet.cpp as a solution that achieves significant speed improvements over existing methods.
- [The Era of 1-bit LLMs:](https://aka.ms/GeneralAI) This paper also discusses the broader context of low-bit LLMs, including the challenges of deploying LLMs on edge and mobile devices and the potential of 1-bit LLMs for this purpose.
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://github.com/microsoft/BitNet/tree/paper) This paper provides detailed technical insights into the implementation of Bitnet.cpp and its performance on various hardware platforms. The paper also discusses the potential of Bitnet.cpp for future hardware optimization.