---
title: Post-training Quantization
category: concepts
created: 2026-04-06T09:29:10.061629+00:00
updated: 2026-04-06T09:29:10.061629+00:00
absorbed: true
---

# Post-training Quantization

## Definition

Post-training quantization is a technique used in machine learning to reduce the precision of model weights and activations, enabling faster inference and lower memory usage without retraining the model. It involves converting high-precision floating-point values (e.g., FP16 or BF16) to lower-bit representations (e.g., 4-bit or 1-bit) to achieve significant reductions in computational and memory costs while maintaining acceptable model performance.

## Context

Post-training quantization is a critical component in the deployment of large language models (LLMs) and other deep learning models. As models grow in size and complexity, the computational and memory demands of inference become increasingly challenging. Post-training quantization addresses these challenges by reducing the bitwidth of model parameters, making inference more efficient and feasible on resource-constrained hardware.

## Key Claims

- **Efficiency**: Post-training quantization significantly reduces memory usage and computational requirements, enabling faster inference on devices with limited resources.
- **Accuracy**: While quantization can introduce some accuracy loss, advanced techniques like GPTQ (Generalized Post-Training Quantization) achieve minimal accuracy degradation, preserving performance metrics like perplexity.
- **Scalability**: Techniques such as GPTQ can compress large models (e.g., 175 billion parameters) to 3-4 bits per parameter, allowing them to run efficiently on a single GPU.
- **Hardware Compatibility**: Quantized models are optimized for hardware that supports low-bit arithmetic, enabling deployment on edge devices and cloud infrastructure.

## Connections

Post-training quantization is closely related to other model compression techniques, including:

- **Quantization-Aware Training (QAT)**: Unlike post-training quantization, QAT incorporates quantization into the training process to mitigate accuracy loss.
- **Model Pruning**: Pruning removes redundant parameters, often used in conjunction with quantization to further reduce model size.
- **Layer-Wise Quantization**: This approach quantizes individual layers or blocks of layers, which can be more efficient than global quantization.
- **Activation Quantization**: This technique quantizes activations rather than weights, which can be more effective in certain scenarios.

Post-training quantization is also connected to the broader field of **neural network inference optimization**, which includes techniques like **pruning**, **quantization**, and **compression** to improve the efficiency and performance of deep learning models.

## Sources

- **awq-2306.00978**: Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. "Training verifiers to solve math word problems," 2021.
- **Dettmers, T. and Zettlemoyer, L.**: "The case for 4-bit pre-precision: k-bit inference scaling laws," arXiv preprint arXiv:2212.09720, 2022.
- **Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L.**: "Llm.int8(): 8-bit matrix multiplication for transformers at scale," arXiv preprint arXiv:2208.07339, 2022.
- **The Era of 1-bit LLMs**: Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 bits," arXiv preprint arXiv:2402.17764, 2024.
- **GPTQ**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. "GPTQ: Accurate post-training quantization for generative pre-trained transformers," arXiv preprint arXiv:2210.17323, 2023.
- **BitNet**: Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. "BitNet: Scaling 1-bit transformers for large language models," arXiv preprint arXiv:2310.11453, 2023.
- **SmoothQuant**: Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. "SmoothQuant: Accurate and efficient post-training quantization for large language models," in Proceedings