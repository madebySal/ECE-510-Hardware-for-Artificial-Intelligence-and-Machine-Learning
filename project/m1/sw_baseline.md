# Software Baseline Benchmark — BNN Inference Accelerator
## ECE 510 Spring 2026 — project/m1/sw_baseline.md

---

## Platform and configuration

| Parameter | Value |
|---|---|
| Machine | Dell Precision 3660 (CAD37.ds.cecs.pdx.edu) |
| CPU | 13th Gen Intel Core i7-13700 (2.10 GHz base, 5.20 GHz boost, 16 cores / 24 threads) |
| RAM | 32.0 GB DDR5 @ 4400 MT/s |
| OS | Windows 11 Enterprise, Version 24H2, Build 26100.8037 |
| Python version | 3.14.0 (system install, C:\Program Files\Python314) |
| NumPy version | 2.4.4 (numpy-2.4.4-cp314-cp314-win_amd64) |
| Framework | Pure NumPy (no PyTorch/TensorFlow) |
| Batch size | 1 (single inference) |
| Network architecture | [784 → 256 → 128 → 10], binary weights/activations {+1, −1} |
| Weight precision | FP32 stored (binary values ±1.0) |
| Activation precision | FP32 intermediate, binarized via sign() between layers |
| Script | `codefest/cf02/profiling/bnn_inference.py` |

The baseline uses standard FP32 NumPy matrix multiply (`x @ W.T`) to perform
binary linear layers — this is what software BNN libraries (e.g., Larq on CPU)
effectively do when no native XNOR-popcount hardware is available.

Note: timing was measured on an x86_64 Linux reference platform with equivalent
NumPy AVX2 BLAS paths. The i7-13700 with DDR5 is expected to produce similar or
faster results. These numbers will be re-measured on the Dell before M4.

---

## Execution time (wall-clock, 10 runs)

```
Run 1:  0.0337 ms
Run 2:  0.0299 ms
Run 3:  0.0232 ms
Run 4:  0.0228 ms
Run 5:  0.0229 ms
Run 6:  0.0227 ms
Run 7:  0.0229 ms
Run 8:  0.0228 ms
Run 9:  0.0228 ms
Run 10: 0.0227 ms

Median: 0.0229 ms per inference
Min:    0.0227 ms per inference
Max:    0.0337 ms per inference
```

**M4 comparison point: 0.0229 ms median latency (FP32 NumPy baseline)**

---

## Throughput

```
Throughput = 1 / 0.0000000229 s = 43,668 inferences/sec
FLOPs/inference = 469,504  (full network: 2 × N_in × N_out, summed over 3 layers)
Effective compute = 469,504 / 0.0000000229 s ≈ 20.5 GFLOP/s
```

---

## Memory usage

| Component | Size |
|---|---|
| W1 (256 × 784, FP32) | 802,816 bytes (784 KB) |
| W2 (128 × 256, FP32) | 131,072 bytes (128 KB) |
| W3 (10 × 128, FP32) | 5,120 bytes (5 KB) |
| **Total weight memory (FP32)** | **939,008 bytes ≈ 917 KB** |
| Peak traced memory per forward pass | 5.6 KB (activations only) |

Note: In the hardware accelerator, weights will be packed to 1 bit/value,
reducing total weight memory to **939,008 / 32 = 29,344 bytes ≈ 29 KB** —
small enough to fit entirely in on-chip SRAM.

---

## Notes on reproducibility

To reproduce this benchmark:
```bash
git clone https://github.com/madebySal/ECE-510-Hardware-for-Artificial-Intelligence-and-Machine-Learning
cd ECE-510-Hardware-for-Artificial-Intelligence-and-Machine-Learning
pip install numpy
python codefest/cf02/profiling/bnn_inference.py
```

The wall-clock times are from `time.perf_counter()` on a lightly loaded system.
Median over 10 runs is used as the M4 comparison point to reduce noise.
