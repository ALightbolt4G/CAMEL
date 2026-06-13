# Training Hardware Specifications

This document outlines the hardware and environment used for training the CAMEL MSLM Architecture. A primary goal of the CAMEL project is to prove that highly capable, multi-domain AI systems can be trained and run on consumer-grade, resource-constrained hardware.

## Hardware Specs
*   **Machine:** Dell Precision 7520
*   **CPU:** Intel Core i7-7820HQ (4 cores, 8 threads, 3.9GHz)
*   **GPU:** NVIDIA Quadro M1200
*   **VRAM:** 4GB
*   **RAM:** 32GB DDR4

## Software Environment
*   **OS:** Windows with WSL2 (Ubuntu)
*   **CUDA:** 12.8 / Driver on Windows
*   **PyTorch:** 2.4.0 cu121

## The "4GB VRAM" Challenge
Why this hardware? Most modern LLMs require massive datacenter GPUs to train. By restricting the CAMEL architecture to a 4GB Quadro M1200, we enforce strict memory efficiency:
1.  **Base Model Quantization:** The Base Model uses `bigscience/bloom-560m`. This is a crucial architectural decision: The Base Cell is responsible for **language understanding only** (hence using a multilingual, 560M parameter model with an Apache 2.0 license). All domain-specific knowledge is the responsibility of the Specialist Cells. It is loaded in 4-bit precision using `bitsandbytes`, taking < 1GB of VRAM.
2.  **LoRA Cells:** Each specialized cell is trained as a Low-Rank Adaptation (LoRA) adapter, taking merely ~10MB of storage and minimal VRAM overhead during dynamic loading.
3.  **Sparse Activation:** The Biological Router ensures only the necessary LoRA adapters are loaded or merged during inference, guaranteeing the 4GB ceiling is never breached.
