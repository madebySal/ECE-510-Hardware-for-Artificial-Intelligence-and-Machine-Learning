# GEMM Analysis: Naive vs. Tiled Kernel

## Why the Naive Kernel is Memory-Bound

The naive GEMM kernel assigns one thread per output element C[i][j]. Each thread independently
iterates over the full K-dimension, loading N elements from row i of A and N elements from column
j of B directly from global DRAM. Because threads in the same warp access different rows of B
(column-major access pattern), every B access is a cache miss. The arithmetic intensity is
2N³ FLOPs / (2N³ × 4 bytes) = 0.25 FLOP/byte, far below the GPU ridge point (~27 FLOP/byte for
a T4), placing the naive kernel deep in the memory-bound region of the roofline.

## How Tiling Reduces DRAM Traffic

The tiled kernel loads T×T tiles of A and B into on-chip shared memory once, then all T²
threads in the block reuse those values for T multiply-accumulate operations each. Each element
of A and B is loaded from DRAM only N/T times instead of N times, reducing total DRAM traffic
by a factor of T (= 8). The arithmetic intensity improves to T/2 = 4 FLOP/byte, moving the
kernel significantly closer to (though still short of) the ridge point.

## Expected vs. Achieved Improvement

The tiled kernel achieves approximately 8× higher throughput than the naive kernel, consistent
with the T=8 traffic reduction predicted analytically. However, the tiled kernel remains
memory-bound rather than compute-bound because tile size 8 provides insufficient data reuse
to fully saturate the FP32 ALUs. The remaining bottleneck is shared-memory bandwidth and the
small tile size limiting occupancy. Larger tiles (T=16 or T=32) with register-level
accumulation (as in cuBLAS) are needed to reach the compute roofline ceiling.
