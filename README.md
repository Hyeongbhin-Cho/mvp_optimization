# MVP: CUDA Optimization for View-Dependent Opacity

This repository contains the **optimized CUDA implementation** for applying Spherical Harmonics (SH) to Opacity, a key component of the **MVP (Multiview Projection)** model.

Unlike standard 3D Gaussian Splatting which uses scalar opacity, this module calculates **View-Dependent Opacity** to handle complex scene attributes. To achieve high performance and memory efficiency, we implemented custom CUDA kernels.

### Key Features
* **View-Dependent Opacity:** Computes opacity dynamically based on viewing direction using Spherical Harmonics.
* **High Performance:** ~2.6x faster than native PyTorch implementation.
* **Memory Efficient:** Uses a **Recomputation Strategy** during the backward pass (instead of saving large activation tensors) to significantly reduce VRAM usage.
* **Kernel Fusion:** Fuses Normalization, SH calculation, and Sigmoid activation into a single kernel.

---

## Installation

Ensure you have **PyTorch** and **CUDA Toolkit** installed before proceeding.

### 1. Install Dependencies (`gsplat`)
This project relies on the ecosystem of 3D Gaussian Splatting. Please install `gsplat` first.

```bash
# Install gsplat (Check standard installation guide for your specific CUDA version)
pip install gsplat
```

### 2. Install MVP CUDA Extension
```bash
cd MVP/cuda

pip install -v -e . --no-build-isolation
```
