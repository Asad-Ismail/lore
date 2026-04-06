---
title: GPTQ
category: techniques
created: 2026-04-06T09:18:39.310340+00:00
updated: 2026-04-06T09:18:39.310340+00:00
absorbed: true
---

## Definition

GPTQ (GPT Quantization) is a post-training quantization method for large language models (LLMs) that uses second-order information to perform error compensation during quantization. It aims to reduce the memory footprint and improve the inference speed of LLMs while maintaining their performance.

## Context

GPTQ is a quantization algorithm developed to address the high memory and computational costs associated with large language models. As LLMs grow in size, the memory requirements for inference become increasingly prohibitive, especially for deployment on edge devices and low-end hardware. GPTQ is designed to reduce the memory footprint of LLMs by quantizing their weights to lower bit precision, such as 4-bit or 8-bit integers, while maintaining the model's performance. This makes it possible to deploy large LLMs on devices with limited memory and computational resources.

## Key Claims

- GPTQ is a post-training quantization method that uses second-order information to perform error compensation during quantization.
- It is designed to reduce the memory footprint and improve the inference speed of LLMs while maintaining their performance.
- GPTQ has been shown to outperform other quantization methods in terms of performance and efficiency.
- It is particularly effective for large LLMs, such as GPT-3 and BERT, and can be applied to a wide range of models and tasks.
- GPTQ has been adopted by several industry and open-source communities, including HuggingFace Transformers, NVIDIA TensorRT-LLM, Microsoft DirectML, Google Vertex AI, Intel Neural Compressor, Amazon Sagemaker, AMD, FastChat, vLLM, and LMDeploy.

## Connections

GPTQ is related to other quantization methods such as AWQ (Activation-aware Weight Quantization), which is another post-training quantization method that uses activation-aware techniques to improve quantization performance. GPTQ is also related to other quantization-aware training (QAT) methods, which are pre-training methods that incorporate quantization during the training process. However, GPTQ is a post-training method that does not require any training or reconstruction, making it more efficient and easier to implement.

## Sources

- [awq-2306.00978, chunk 1]: This excerpt introduces the concept of GPTQ and its application in reducing the memory footprint of LLMs. It also discusses the limitations of existing quantization methods and how GPTQ addresses them.
- [awq-2306.00978, chunk 2]: This excerpt provides a detailed description of GPTQ, including its methodology, implementation, and performance results. It also compares GPTQ with other quantization methods and discusses its advantages and limitations.
- [awq-2306.00978, chunk 7]: This excerpt discusses the implementation details of GPTQ, including the use of INT3 kernels for OPT models and GPTQ-for-LLaMA extensions for INT4 reordered quantization. It also describes the system support for low-bit quantized LLMs and the performance results of GPTQ on various models and tasks.
- [awq-2306.00978, chunk 8]: This excerpt provides a detailed description of the performance results of GPTQ on various models and tasks, including the comparison with other quantization methods such as AWQ and RTN. It also discusses the generalization of GPTQ to instruction-tuned models and multi-modal LMs.
- [awq-2306.00978, chunk 9]: This excerpt discusses the results of GPTQ on various models and tasks, including the comparison with other quantization methods and the performance results on different bit precisions. It also discusses the advantages of GPTQ in terms of memory efficiency and inference speed.
- [awq-2306.00978, chunk 10]: This excerpt provides a detailed description of the performance results of GPTQ on various models and tasks, including the comparison with other quantization methods and the performance results on different bit precisions. It also discusses the advantages of GPTQ in terms of memory efficiency and inference speed.
- [awq-2306.00978, chunk 11]: This excerpt discusses the results of GPTQ on various models and tasks, including the comparison with other quantization methods and the performance results on different bit precisions. It also discusses the advantages of GPTQ in terms of memory efficiency and inference speed.
- [awq-2306.00978, chunk 12]: This excerpt provides a detailed description of the performance results of GPTQ on various models and tasks, including the comparison with other quantization methods and the performance results on different bit precisions. It also discusses the advantages of GPTQ in terms of memory