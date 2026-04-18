# GEMM Analysis: Naive vs. Tiled Kernel (GTX 1650, 1024×1024 FP32)

## Measured Performance

| Kernel | Avg Time | Throughput |
|--------|----------|------------|
| Naive  | 9.87 ms  | 217.65 GFLOP/s |
| Tiled (T=8) | 10.32 ms | 208.13 GFLOP/s |

GPU: NVIDIA GeForce GTX 1650 — Peak FP32: 2984 GFLOP/s, Peak BW: 128 GB/s, Ridge: ~23.3 FLOP/byte.
Nsight Compute hardware counters unavailable on Windows without admin rights (ERR_NVGPUCTRPERM);
timing measured via cudaEventRecord over 10 runs.

## Why the Naive Kernel is Memory-Bound

The naive kernel assigns one thread per output element C[i][j]. Each thread independently loads
N=1024 values from a row of A and N values from a column of B directly from global memory, with
no data sharing between threads. Column accesses into B are non-coalesced (stride-N), causing
one cache-line fetch per element. The theoretical arithmetic intensity is only 0.25 FLOP/byte,
far below the GTX 1650 ridge point of 23.3 FLOP/byte, placing it firmly in the memory-bound
region of the roofline.

## How Tiling Reduces DRAM Traffic

The tiled kernel loads T×T blocks of A and B into on-chip shared memory once, then all T²
threads in the block reuse those values for T dot-product steps before fetching the next tile.
Each A and B element is fetched from DRAM only N/T = 128 times instead of N = 1024 times in
the naive case, reducing theoretical DRAM traffic by a factor of T = 8 and raising arithmetic
intensity to T/2 = 4 FLOP/byte.

## Expected vs. Achieved Improvement

The tiled kernel achieved 208 GFLOP/s versus 217 GFLOP/s for naive — virtually no improvement,
and both well below the 2984 GFLOP/s compute ceiling. Both kernels land in the memory-bound
region of the roofline. The reason the improvement is smaller than expected is twofold: first,
the GTX 1650's L2 cache (1 MB) partially absorbs the repeated B accesses in the naive kernel,
masking the true DRAM penalty. Second, tile size T=8 yields only 64-thread thread blocks, which
is too small to hide memory latency and achieve high occupancy on the SM. The remaining
bottleneck is low thread-level parallelism and insufficient data reuse per shared-memory load.
Larger tiles (T=16 or T=32) with register-level accumulation are needed to approach the
compute ceiling.
