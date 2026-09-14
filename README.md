# CUDA Worklog

Kernel-by-kernel practice log. Every kernel written from scratch, measured, and profiled.

## Hardware

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| Compute capability | 8.0 (`-arch=sm_80`) |
| Global memory | 80 GB HBM2e |
| SMs | 108 (6912 CUDA cores) |
| SM clock | 1410 MHz |
| Memory clock | 1512 MHz |
| Memory bus | 5120-bit |
| L2 cache | 40 MB |
| Shared memory / block | 48 KB (164 KB per SM) |
| Max threads / block | 1024 |
| Max threads / SM | 2048 |
| **Peak memory bandwidth** | **1935 GB/s** |

## Results

| Kernel | ms | GB/s | % of peak | Notes |
|---|---|---|---|---|

