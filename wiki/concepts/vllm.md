---
title: vLLM
category: concepts
created: 2026-04-06T09:36:49.247640+00:00
updated: 2026-04-06T09:36:49.247640+00:00
absorbed: true
---

## Definition  
vLLM is a high-performance inference framework designed for large language models (LLMs), enabling efficient deployment of quantized and low-bit models on a wide range of hardware, including edge devices and GPUs. It supports hardware-friendly low-bit weight-only quantization, significantly reducing memory footprint and accelerating inference while maintaining high accuracy.

## Context  
vLLM emerged as a response to the challenges of deploying large-scale LLMs on resource-constrained hardware. The increasing size of LLMs, such as GPT-3 with 175B parameters, necessitates efficient inference methods to reduce memory and computational costs. Traditional approaches like quantization-aware training (QAT) and post-training quantization (PTQ) face limitations in accuracy and efficiency, prompting the development of novel quantization techniques. AWQ (Activation-aware Weight Quantization), introduced in the context of vLLM, addresses these challenges by leveraging activation distributions to identify salient weight channels, thereby minimizing quantization loss. vLLM also incorporates efficient inference frameworks, such as TinyChat, which optimize memory usage and speed through on-the-fly dequantization and kernel fusion, making it possible to deploy large models like Llama-2-70B on devices with limited memory.

## Key Claims  
- vLLM supports hardware-friendly low-bit weight-only quantization, reducing memory footprint and accelerating inference.  
- AWQ, a key component of vLLM, uses activation distributions to identify salient weight channels, minimizing quantization loss and preserving model accuracy.  
- vLLM's TinyChat framework achieves significant speedups on various hardware, including desktops, laptops, and edge devices, with an average 3.2-3.3× speedup compared to FP16 implementations.  
- vLLM enables deployment of large models like Llama-2-70B on devices with limited memory, such as the NVIDIA Jetson Orin with 64GB of memory and a laptop RTX 4070 GPU with 8GB of memory.  
- vLLM has been widely adopted by industry and open-source communities, including HuggingFace Transformers, NVIDIA TensorRT-LLM, Microsoft DirectML, Google Vertex AI, Intel Neural Compressor, Amazon Sagemaker, AMD, FastChat, and LMDeploy.

## Connections  
vLLM is closely related to other LLM inference frameworks and quantization techniques. It builds upon the principles of AWQ, which is detailed in the paper "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration." vLLM also shares similarities with BitNet b1.58, a 1-bit LLM variant that demonstrates superior performance and efficiency. Additionally, vLLM is part of a broader trend in LLM inference optimization, including the use of FlashAttention for GPU attention kernel design and the development of efficient inference techniques like TensorRT-LLM and Powerinfer. The framework is also connected to the broader field of LLM quantization, where post-training quantization (PTQ) and quantization-aware training (QAT) are used to reduce model size and improve inference speed.

## Sources  
- [awq-2306.00978](https://arxiv.org/abs/2306.00978): "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"  
- [The Era of 1-bit LLMs:](https://arxiv.org/abs/2305.15206): "The Era of 1-bit LLMs: BitNet b1.58 and the Future of Efficient Language Models"  
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://github.com/ggerganov/llama.cpp): "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs"  
- [LLM Inference FlashAttention](https://arxiv.org/abs/2205.11919): "FlashAttention: Fast and Memory-Efficient Self-Attention"  
- [VLLM (Kwon et al., 2023)](https://arxiv.org/abs/2305.13561): "vLLM: A High-Performance Inference Framework for Large Language Models"