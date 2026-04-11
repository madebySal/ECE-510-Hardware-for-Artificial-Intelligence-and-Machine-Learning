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
| Python version | **[fill in: run `python --version` on your machine]** |
| NumPy version | **[fill in: run `python -c "import numpy; print(numpy.__version__)"` ]** |
| Framework | Pure NumPy (no PyTorch/TensorFlow) |
| Batch size | 1 (single inference) |
| Network architecture | [784 → 256 → 128 → 10], binary weights/activations {+1, −1} |
| Weight precision | FP32 stored (binary values ±1.0) |
| Activation precision | FP32 intermediate, binarized via sign() between layers |
| Script | `codefest/cf02/profiling/bnn_inference.py` |

The baseline uses standard FP32 NumPy matrix multiply (`x @ W.T`) to perform
binary linear layers — this is what software BNN libraries (e.g., Larq on CPU)
effectively do when no native XNOR-popcount hardware is available.

---

## Execution time (wall-clock, 10 runs)

**⚠️ ACTION REQUIRED: Run `bnn_inference.py` on your machine and replace the timing numbers below with your actual results before submitting.**

The numbers below are placeholder values measured in a build container (x86_64 Linux). The M4 speedup comparison will be against YOUR machine (Dell Precision 3660, i7-13700), so these must be your actual measured values.

```
Run 1:  [replace]
Run 2:  [replace]
Run 3:  [replace]
Run 4:  [replace]
Run 5:  [replace]
Run 6:  [replace]
Run 7:  [replace]
Run 8:  [replace]
Run 9:  [replace]
Run 10: [replace]

Median: [replace] ms per inference
Min:    [replace] ms per inference
Max:    [replace] ms per inference
```

**M4 comparison point: [replace] ms median latency (FP32 NumPy baseline, Dell Precision 3660 i7-13700)**

---

## Throughput

```
Throughput = 1 / [median_s] = [replace] inferences/sec
FLOPs/inference = 469,504  (full network: 2 × N_in × N_out, summed over 3 layers)
Effective compute = 469,504 / [median_s] ≈ [replace] GFLOP/s
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
python3 codefest/cf02/profiling/bnn_inference.py
```

The wall-clock times reported are from `time.perf_counter()` on a lightly loaded
system. Median over 10 runs is used as the M4 comparison point to reduce noise.
