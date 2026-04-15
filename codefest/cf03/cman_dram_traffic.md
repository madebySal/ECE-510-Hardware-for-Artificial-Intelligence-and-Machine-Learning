# CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply

**Given:** N = 32, T = 8, FP32 (4 bytes/element), DRAM BW = 320 GB/s, Compute = 10 TFLOPS

---

## Task 1: Naive Triple Loop

For each output element C[i][j]:

$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]$$

Each element B[k][j] is needed for every row i of C → **each element of B is accessed N = 32 times**.

| Matrix | Access pattern | Total element accesses |
|--------|---------------|----------------------|
| A | Each row i loaded once per j | $N^3 = 32{,}768$ |
| B | Each column j loaded once per i | $N^3 = 32{,}768$ |

**Total naive DRAM traffic:**

$$\text{Naive Traffic} = 2 \times N^3 \times 4\ \text{bytes} = 2 \times 32{,}768 \times 4 = \boxed{262{,}144\ \text{bytes} = 256\ \text{KB}}$$

---

## Task 2: Tiled Loop (T = 8)

The computation is blocked into T×T = 8×8 tiles. Tiles per dimension: N/T = 32/8 = 4.

For each of the $(N/T)^2 = 16$ output tiles of C, we step through N/T = 4 tile pairs along the K dimension. At each step, one T×T tile of A and one T×T tile of B are loaded from DRAM into shared memory and reused T times before the next fetch.

**Total elements loaded from DRAM:**

| Matrix | Tile steps per output tile | Output tiles | Total elements | Bytes |
|--------|---------------------------|--------------|----------------|-------|
| A | $N/T = 4$ | $(N/T)^2 = 16$ | $N^3/T = 4{,}096$ | $16{,}384$ |
| B | $N/T = 4$ | $(N/T)^2 = 16$ | $N^3/T = 4{,}096$ | $16{,}384$ |

$$\text{Tiled Traffic} = 2 \times \frac{N^3}{T} \times 4\ \text{bytes} = 2 \times 4{,}096 \times 4 = \boxed{32{,}768\ \text{bytes} = 32\ \text{KB}}$$

---

## Task 3: Ratio of Naive to Tiled Traffic

$$\text{Ratio} = \frac{2N^3 \times 4}{2(N^3/T) \times 4} = \frac{N^3}{N^3/T} = \boxed{T = 8\times}$$

**One-sentence explanation:**

> Each T×T tile of A and B is loaded from DRAM once and reused T times across the T dot-product steps within the tile, so total DRAM traffic drops by exactly a factor of T.

---

## Task 4: Execution Time & Bottleneck Analysis

**Total FLOPs:** $2N^3 = 2 \times 32{,}768 = 65{,}536$ FLOPs

### Naive Case

$$t_{\text{compute}} = \frac{65{,}536}{10 \times 10^{12}} \approx 6.6\ \text{ns}$$

$$t_{\text{memory}} = \frac{262{,}144}{320 \times 10^9} \approx 819\ \text{ns}$$

**Bottleneck: Memory** ($819\ \text{ns} \gg 6.6\ \text{ns}$) → execution time ≈ **819 ns** → **memory-bound**

### Tiled Case

$$t_{\text{compute}} = \frac{65{,}536}{10 \times 10^{12}} \approx 6.6\ \text{ns}$$

$$t_{\text{memory}} = \frac{32{,}768}{320 \times 10^9} \approx 102\ \text{ns}$$

**Bottleneck: Memory** ($102\ \text{ns} > 6.6\ \text{ns}$) → execution time ≈ **102 ns** → **memory-bound (closer to ridge point)**

### Summary

| Metric | Naive | Tiled (T=8) | Improvement |
|--------|-------|-------------|-------------|
| DRAM traffic | 256 KB | 32 KB | 8× = T |
| Memory time | ~819 ns | ~102 ns | 8× |
| Compute time | ~6.6 ns | ~6.6 ns | 1× |
| Bottleneck | Memory-bound | Memory-bound (closer to ridge) | — |
