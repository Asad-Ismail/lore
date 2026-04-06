---
title: AWQ
category: concepts
created: 2026-04-06T09:17:15.109235+00:00
updated: 2026-04-06T09:17:15.109235+00:00
absorbed: true
---

**AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration**

**Definition**  
AWQ (Activation-Aware Weight Quantization) is a hardware-friendly low-bit weight-only quantization method for large language models (LLMs) that leverages activation distribution to identify and protect salient weight channels, reducing quantization error and improving inference performance on edge devices.

**Context**  
As large language models (LLMs) become increasingly prevalent, deploying them on edge devices is crucial for reducing cloud computing costs and enhancing user privacy. However, the large model sizes and limited hardware resources pose significant challenges. AWQ addresses these challenges by focusing on weight-only quantization, where only the weights are quantized, while activations remain in higher precision. This approach minimizes the memory footprint and accelerates inference, making it suitable for on-device deployment.

**Key Claims**  
- **Activation-aware selection**: AWQ identifies salient weight channels based on activation distribution rather than weight distribution, leading to significant reductions in quantization error.
- **Hardware efficiency**: AWQ avoids mixed-precision quantization by using per-channel scaling to protect salient weights, resulting in better hardware efficiency.
- **Generalization**: AWQ does not rely on backpropagation or reconstruction, allowing it to generalize well across different domains and modalities without overfitting the calibration set.
- **Performance improvements**: AWQ outperforms existing methods on various benchmarks, including language modeling and domain-specific tasks, achieving excellent quantization performance for instruction-tuned and multi-modal LLMs.
- **Speedup**: The accompanying inference framework, TinyChat, offers more than 3× speedup over FP16 implementations on both desktop and mobile GPUs.

**Connections**  
- **Related Concepts**: AWQ is related to other quantization techniques such as GPTQ (Frantar et al., 2022) and W8A8 quantization. However, AWQ's activation-aware approach distinguishes it by focusing on salient weights.
- **Technical Foundations**: AWQ builds on principles from quantization-aware training (QAT) and post-training quantization (PTQ), but it avoids the inefficiencies of mixed-precision quantization through per-channel scaling.
- **Applications**: AWQ is particularly useful for on-device deployment of LLMs, enabling efficient inference on edge devices with limited memory and computational resources.

**Sources**  
- **awq-2306.00978**: The original paper introducing AWQ, which details the activation-aware approach, per-channel scaling, and the accompanying inference framework TinyChat. It provides extensive experimental results showing AWQ's performance improvements over existing methods.
- **awq-2306.00978**: The paper also includes a detailed analysis of the quantization error reduction through activation-aware scaling and discusses the benefits of AWQ for multi-modal LLMs.
- **awq-2306.00978**: The paper provides a comprehensive evaluation of AWQ on various benchmarks, including language modeling, programming, and math tasks, demonstrating its effectiveness across different model sizes and architectures.
- **awq-2306.00978**: The paper includes a discussion on the data efficiency and generalization of AWQ, showing that it requires a smaller calibration set and is robust to different distribution settings. It also highlights the practical deployment of AWQ on edge devices with limited memory.