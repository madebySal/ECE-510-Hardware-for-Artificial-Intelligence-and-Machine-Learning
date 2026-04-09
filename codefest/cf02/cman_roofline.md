# CMAN — Roofline Construction and Kernel Classification

## Hardware specification
- Peak compute: 10 TFLOPS FP32 = 10,000 GFLOP/s
- Peak DRAM bandwidth: 320 GB/s
- Ridge point: 10,000 / 320 = **31.25 FLOP/byte**

---

## Roofline diagram

---

## Kernel A — Dense GEMM (1024×1024 FP32)

**FLOPs:**  
FLOPs = 2 × N³ = 2 × 1024³ = **2,147,483,648 FLOPs ≈ 2.147 GFLOPs**

**Bytes transferred (no cache reuse):**  
- Matrix A: 1024 × 1024 × 4 = 4,194,304 bytes  
- Matrix B: 1024 × 1024 × 4 = 4,194,304 bytes  
- Matrix C (written): 1024 × 1024 × 4 = 4,194,304 bytes  
- **Total = 12,582,912 bytes ≈ 12 MB**

**Arithmetic Intensity:**  
AI = 2,147,483,648 / 12,582,912 = **170.67 FLOP/byte**

**Classification:** 170.67 > 31.25 → **Compute-bound**

**Attainable performance:**  
min(10,000, 320 × 170.67) = min(10,000, 54,614) = **10,000 GFLOP/s** (hits compute ceiling)

**Architectural recommendation:** GEMM is compute-bound, so the bottleneck is ALU throughput, not bandwidth. The most impactful change would be adding more FP32 compute units (wider SIMD, more cores) or using matrix-multiply accelerators (e.g., tensor cores), since adding memory bandwidth would not improve performance.

---

## Kernel B — Vector addition (N = 4,194,304 FP32)

**FLOPs:**  
FLOPs = N × 1 = **4,194,304 FLOPs ≈ 4.19 MFLOPs**

**Bytes transferred (no cache reuse):**  
- Vector A: 4,194,304 × 4 = 16,777,216 bytes  
- Vector B: 4,194,304 × 4 = 16,777,216 bytes  
- Vector C (written): 4,194,304 × 4 = 16,777,216 bytes  
- **Total = 50,331,648 bytes ≈ 48 MB**

**Arithmetic Intensity:**  
AI = 4,194,304 / 50,331,648 = **0.0833 FLOP/byte**

**Classification:** 0.0833 << 31.25 → **Memory-bound**

**Attainable performance:**  
min(10,000, 320 × 0.0833) = min(10,000, 26.67) = **26.67 GFLOP/s** (far below compute ceiling)

**Architectural recommendation:** Vector-add is deeply memory-bound with essentially one FLOP per 12 bytes of traffic. Adding more compute does nothing — the only impactful change is higher memory bandwidth (e.g., HBM over GDDR) or eliminating the DRAM traffic altogether via in-memory/near-memory computation.
