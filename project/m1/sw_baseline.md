# Software Baseline Benchmark – BNN Inference
## ECE 510 – Milestone 1 | Spring 2026

---

## Platform Configuration

| Field | Value |
|-------|-------|
| Machine | Dell Precision 3660 (CAD37.ds.cecs.pdx.edu) |
| CPU | 13th Gen Intel Core i7-13700 (2.10 GHz base, 5.20 GHz boost, 16 cores / 24 threads) |
| RAM | 32.0 GB DDR5 @ 4400 MT/s |
| OS | Windows 11 Enterprise, Version 24H2, Build 26100.8037 |
| Python | 3.14.0 |
| NumPy | 2.4.4 |
| PyTorch | 2.2.0 (CPU only, used for ResNet-18 comparison) |
| Batch size | 1 (single-sample inference) |

---

## BNN Software Baseline

**Architecture:** Fully-connected BNN [784 → 256 → 128 → 10]  
**Weights:** Binary (+1/−1), stored as FP32 in software (NumPy float32)  
**Input:** Random FP32 vector, shape [1, 784]  
**Runs:** 20 forward passes (5 warm-up, 20 timed)

| Metric | Value |
|--------|-------|
| Median time per inference | 149.42 µs |
| Throughput | 6,692.5 samples/sec |
| Peak memory (RSS) | 6.4 KB |

---

## Dominant Kernel

The dominant kernel is the Layer 1 matrix multiply `[1×784] @ [784×256]` implemented via
`numpy.matmul`, accounting for **>80% of total forward-pass runtime** (cProfile).
This operation accounts for the largest share of compute (FLOPs and memory
traffic) in the forward pass. See `codefest/cf02/analysis/ai_calculation.md` for full details.

---

## Reproducibility

To reproduce this benchmark:
```bash
python - << 'EOF'
import numpy as np, time, tracemalloc

np.random.seed(42)
W1 = np.sign(np.random.randn(784, 256)).astype(np.float32)
W2 = np.sign(np.random.randn(256, 128)).astype(np.float32)
W3 = np.sign(np.random.randn(128, 10)).astype(np.float32)

def binarize(x): return np.sign(x).astype(np.float32)
def bnn_forward(x):
    x = binarize(x); x = x @ W1
    x = binarize(x); x = x @ W2
    x = binarize(x); x = x @ W3
    return x

x_in = np.random.randn(1, 784).astype(np.float32)
for _ in range(5): bnn_forward(x_in)  # warm-up

RUNS = 20
tracemalloc.start()
t0 = time.perf_counter()
for _ in range(RUNS): bnn_forward(x_in)
t1 = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

per_pass = (t1 - t0) / RUNS
print(f"Time/inference: {per_pass*1e6:.2f} us")
print(f"Throughput:     {1/per_pass:.1f} samples/sec")
print(f"Peak memory:    {peak_mem/1024:.1f} KB")
EOF
```
