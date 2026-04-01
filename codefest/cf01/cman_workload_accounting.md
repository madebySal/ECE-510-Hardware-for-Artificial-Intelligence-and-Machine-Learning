# CF01 – Workload Accounting: Binary Neural Network Inference

## Target Kernel
Binary Neural Network (BNN) fully-connected layer using XNOR + popcount operations.

---

## Operation Count

| Operation | Standard NN (FP32) | BNN (Binary) |
|---|---|---|
| Multiply | 1 FP32 MUL per weight | 1 XNOR (1 bit) |
| Accumulate | 1 FP32 ADD per weight | 1 popcount increment |
| Precision | 32-bit | 1-bit |

For a layer with input size N and output size M:
- Total operations: N × M XNOR + popcount
- Equivalent FLOPs: 2 × N × M (multiply + add)

Example (N=256, M=128):
- Operations: 256 × 128 = 32,768 XNOR-popcount ops
- Equivalent: ~65,536 FLOPs

---

## Memory Traffic

| Data | Size | Notes |
|---|---|---|
| Input activations | N / 8 bytes | 1-bit packed |
| Binary weights | N × M / 8 bytes | 1-bit packed |
| Output | M × 4 bytes | INT32 accumulator |

Example (N=256, M=128):
- Inputs: 32 bytes
- Weights: 4,096 bytes
- Output: 512 bytes
- **Total memory traffic: ~4,640 bytes**

---

## Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes
                     = 65,536 / 4,640
                     ≈ 14.1 FLOPs/byte
```

This places the kernel in the **compute-bound** region on a typical roofline plot,
making it a good candidate for hardware acceleration.

---

## Summary

- BNN inference replaces FP32 MACs with 1-bit XNOR+popcount
- Memory footprint is ~32× smaller than FP32 equivalent
- Arithmetic intensity ~14 FLOPs/byte → compute-bound
- Hardware accelerator can exploit bitwise parallelism for high throughput
