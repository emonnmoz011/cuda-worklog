## Hardware — lab machine (RTX 3050)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 OEM |
| Compute capability | 8.6 (`-arch=sm_86`) |
| Global memory | 8 GB |
| SMs | 18 |
| Max threads / SM | 1536 (48 warps) |
| Shared memory / block | 48 KB |
| Memory clock | 7001 MHz |
| Memory bus | 128-bit |
| **Peak memory bandwidth** | **224 GB/s** |



## Hardware — personal PC (RTX 5070, WSL2)

| | |
|---|---|
| Compute capability | 12.0 (`-arch=sm_120`) |
| Memory | 12 GB |
| SMs | 48 |
| Max warps / SM | 48 |
| DRAM frequency | 13.99 GHz |
| Peak bandwidth | 671 GB/s |

## Results

| Kernel | GB/s | % of peak | Notes |
|---|---|---|---|
| vector add (256 threads) | 592.5 | 88.3 | memory-bound |

## Block size sweep — vector add, N = 100M

| Threads | GB/s | % of peak |
|---|---|---|
| 64 | 593.9 | 88.5 |
| 128 | 590.4 | 88.0 |
| 256 | 592.5 | 88.3 |
| 512 | 592.8 | 88.3 |
| 1024 | 562.8 | 83.9 |

Block size has almost no effect here. 64 through 512 are within 0.6% of
each other. Only 1024 loses speed: one block is 32 warps, and the SM
allows 48, so just one block fits and a third of the warp slots sit empty.

For a memory-bound kernel the pipe is already near full, so regrouping
threads does not help.