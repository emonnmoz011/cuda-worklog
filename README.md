# CUDA Worklog

Kernel-by-kernel practice. Every kernel written from scratch, measured, and profiled.

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
| **Peak memory bandwidth** | **671 GB/s** |

## Results — RTX 5070

| Kernel | GB/s | % of peak | Notes |
|---|---|---|---|
| vector add | 592.5 | 88.3 | both accesses coalesced |
| SAXPY | 585.3 | 87.2 | same bytes as vector add, confirms the harness |
| array reverse | 584.0 | 87.0 | reversed writes stay inside the same chunk |
| stencil naive | 581.8 | 86.7 | 7 reads per element, L1 absorbs the repeats |
| stencil shared | 574.5 | 85.6 | slightly slower, cache already did the job |
| scatter (stride 8) | 111.2 | 16.6 | 5.3x slower, each thread in its own chunk |
| transpose naive | 220.9 | 32.9 | read coalesced, write scattered |
| transpose shared | 318.0 | 47.4 | shared memory makes both global accesses coalesced |
| transpose padded | 419.7 | 62.5 | tile[32][33] removes bank conflicts, +32% |

## What the numbers taught

**Coalescing dominates.** Stride 8 cost 5.3x. Nothing else came close to
that penalty.

**Shared memory is not always a win.** It lost on the stencil, where L1
already handled the reuse. It won big on the transpose, where the problem
was a scattered write pattern no cache can fix.

**Bank conflicts are cheap to fix.** One character, `tile[32][33]` instead
of `tile[32][32]`, gave 32%.

## Block size sweep — vector add, N = 100M

| Threads | GB/s | % of peak |
|---|---|---|
| 64 | 593.9 | 88.5 |
| 128 | 590.4 | 88.0 |
| 256 | 592.5 | 88.3 |
| 512 | 592.8 | 88.3 |
| 1024 | 562.8 | 83.9 |

Block size has almost no effect here. 64 through 512 sit within 0.6% of
each other. Only 1024 loses speed: one block is 32 warps and the SM allows
48, so just one block fits and a third of the warp slots sit empty.

For a memory-bound kernel the pipe is already near full, so regrouping
threads does not help.
