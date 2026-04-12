# Arithmetic Intensity Calculation – BNN Inference Accelerator
## ECE 510 – CF02 | Spring 2026

**Project:** Binary Neural Network (BNN) Inference Accelerator  
**Architecture:** Fully-connected BNN [784 → 256 → 128 → 10], batch size 1, FP32 software baseline  
**Dominant kernel:** Layer 1 matrix multiply — `[1×784] @ [784×256]` (identified via cProfile as largest share of runtime)

---

## 1. Dominant Kernel Identification

From `project_profile.txt`: the dominant operation is the NumPy matrix multiply (`x @ W1`) in `bnn_forward`,
specifically the Layer 1 matmul `[1×784] @ [784×256]`. This layer contains the largest weight matrix and
dominates runtime for a single forward pass.

**Dominant kernel:** `numpy.core._multiarray_umath.matmul` (Layer 1: [1×784] @ [784×256])

---

## 2. FLOPs Calculation

For a matrix multiply `[1×M] @ [M×N]`, FLOPs = 2 × M × N (one multiply + one accumulate per output element).

| Layer | Formula | M | N | FLOPs |
|-------|---------|---|---|-------|
| Layer 1 | 2 × M × N | 784 | 256 | 2 × 784 × 256 = **401,408** |
| Layer 2 | 2 × M × N | 256 | 128 | 2 × 256 × 128 = 65,536 |
| Layer 3 | 2 × M × N | 128 | 10  | 2 × 128 × 10  = 2,560 |
| **Total** | | | | **469,504 FLOPs ≈ 0.47 MFLOPs** |

**Dominant kernel FLOPs (Layer 1):** 2 × 784 × 256 = **401,408 FLOPs**

---

## 3. Bytes Transferred (DRAM, No Reuse, FP32 = 4 bytes/element)

| Operand | Size | Bytes |
|---------|------|-------|
| Input activation (x) | 1 × 784 × 4 | 3,136 B |
| Weight matrix W1 | 784 × 256 × 4 | 802,816 B |
| Output activation | 1 × 256 × 4 | 1,024 B |
| **Layer 1 total** | | **806,976 B ≈ 0.807 MB** |

**Full forward pass bytes:**

| Operand | Bytes |
|---------|-------|
| W1 (784×256×4) | 802,816 |
| W2 (256×128×4) | 131,072 |
| W3 (128×10×4)  | 5,120 |
| Input + all activations ((784+256+128+10)×4) | 4,712 |
| **Total** | **943,720 B ≈ 0.944 MB** |

---

## 4. Arithmetic Intensity

**Dominant kernel (Layer 1):**
```
AI = FLOPs / Bytes = 401,408 / 806,976 = 0.497 FLOP/byte ≈ 0.50 FLOP/byte
```

**Full forward pass:**
```
AI = 469,504 / 943,720 = 0.497 FLOP/byte ≈ 0.50 FLOP/byte
```

---

## 5. Roofline Position

**Hardware:** Intel Core i7-1165G7  
**Peak compute:** 150 GFLOP/s (FP32, AVX2) — source: Intel ARK  
**Peak memory bandwidth:** 40 GB/s (DDR4-3200 dual-channel) — source: Intel ARK  
**Ridge point:** 150 / 40 = **3.75 FLOP/byte**

**Result:** AI = 0.50 FLOP/byte < ridge point of 3.75 FLOP/byte  
→ BNN inference is **memory-bound** on this CPU.  
→ Attainable performance = 0.50 × 40 GB/s = **20 GFLOP/s** (far below the 150 GFLOP/s compute ceiling).

The kernel sits deep in the memory-bound region. The theoretical ceiling is only 20 GFLOP/s vs 150 GFLOP/s peak — a **7.5× gap** that cannot be closed by adding more FP32 compute units.
