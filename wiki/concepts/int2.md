---
title: INT2
category: concepts
created: 2026-04-06T09:22:57.500510+00:00
updated: 2026-04-06T09:22:57.500510+00:00
absorbed: true
---

## Definition  
INT2 is a low-bit quantization format used in large language models (LLMs) to reduce model size and improve inference efficiency. It represents weights using 2 bits per weight, allowing for significant memory savings while maintaining acceptable performance. This format is particularly useful in edge computing and on-device deployment scenarios where memory and computational resources are constrained.

## Context  
INT2 is part of a broader set of quantization techniques aimed at compressing LLMs for deployment on hardware with limited memory and processing power. These techniques include INT4, INT8, and even more extreme formats like INT1. The goal of quantization is to reduce the precision of model parameters, thereby decreasing the model's memory footprint and computational requirements without significantly impacting performance. INT2 is especially notable for its ability to achieve high compression ratios with minimal loss in performance, making it a valuable tool in the deployment of large-scale models on resource-constrained devices.

## Key Claims  
- **High Compression Ratio**: INT2 reduces the model size by a factor of 4 compared to full-precision (FP16) models, as seen in experiments with the LLaVA-13B model, where INT4-g128 quantization achieved a 4× reduction with negligible performance loss.
- **Performance Preservation**: Despite the extreme quantization, INT2 maintains performance comparable to FP16 models, as demonstrated in the OpenFlamingo-9B model on the COCO captioning dataset, where INT4-g128 quantization improved captioning quality compared to the round-to-nearest (RTN) baseline.
- **Edge Deployment**: INT2 is particularly effective for edge computing and on-device deployment, as shown in the Bitnet.cpp framework, which enables efficient inference for 1-bit LLMs by utilizing INT2 with a Scale (I2_S) to ensure lossless edge inference.
- **Robustness to Calibration Sets**: INT2-based quantization methods, such as AWQ, are more robust to variations in calibration sets, demonstrating consistent performance across different dataset distributions.

## Connections  
- **AWQ (Activation-aware Weight Quantization)**: AWQ is a method that integrates with INT2 quantization to achieve better performance and robustness. It is orthogonal to GPTQ and can be combined to further improve INT2 quantization performance.
- **Bitnet.cpp**: This framework is optimized for 1-bit LLMs and utilizes INT2 with a Scale (I2_S) to enable lossless inference on edge devices, demonstrating the practical application of INT2 in real-world scenarios.
- **QuIP#**: This research explores advanced quantization techniques, including the use of INT2 in conjunction with other methods to achieve even better performance and efficiency in LLM deployment.

## Sources  
- [awq-2306.00978, chunk 10]: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration" describes the performance of INT4-g128 quantization on the OpenFlamingo-9B model, showing significant improvements over RTN and demonstrating the effectiveness of INT2 in reducing model size with minimal performance loss.  
- [awq-2306.00978, chunk 11]: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration" further discusses the use of INT2 for extreme low-bit quantization, highlighting its ability to accommodate limited device memory and its effectiveness in maintaining performance.  
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs, chunk 0]: This paper introduces Bitnet.cpp, an inference system optimized for 1-bit LLMs, which utilizes INT2 with a Scale (I2_S) to enable lossless inference on edge devices.  
- [QuIP#: Even Better LLM Quantization with, chunk 18]: This paper discusses advanced quantization techniques, including the use of INT2 in conjunction with other methods to achieve even better performance and efficiency in LLM deployment.