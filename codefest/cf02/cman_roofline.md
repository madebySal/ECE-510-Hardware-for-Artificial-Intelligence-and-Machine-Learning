# CMAN — Roofline Construction and Kernel Classification
## ECE 510 Spring 2026 — Codefest 2

## Hardware specification
- Peak compute: 10 TFLOPS FP32 = 10,000 GFLOP/s
- Peak DRAM bandwidth: 320 GB/s
- Ridge point: 10,000 / 320 = **31.25 FLOP/byte**

---

## Roofline diagram

```
y: GFLOP/s (log)
|
10,000 |_________________________________________[compute ceiling]________
       |                                    ___/
       |                               ___/ 
       |                          ___/      
       |                     ___/           
 1,000 |                ___/                
       |           ___/     [B] vec-add     [A] GEMM
       |      ___/          (0.083, 26.7)   (170.7, 10000)
       |  ___/              *               
   100 |_/                                   
       |                                    
    10 |                                    
       +--+-------+--------+--------+-------+---> x: FLOP/byte (log)
       0.01      0.1       1       10      100    1000
                                   ^
                              ridge=31.25
```

- **[A] GEMM** at AI = 170.67 FLOP/byte → compute-bound, hits 10,000 GFLOP/s ceiling
- **[B] Vector-add** at AI = 0.0833 FLOP/byte → memory-bound, attainable 26.67 GFLOP/s

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

**Architectural recommendation:** GEMM is compute-bound — the bottleneck is ALU throughput.
Adding more FP32 compute units (wider SIMD, tensor cores) would improve performance; adding
memory bandwidth would not.

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
min(10,000, 320 × 0.0833) = min(10,000, 26.67) = **26.67 GFLOP/s**

**Architectural recommendation:** Vector-add is deeply memory-bound — adding more compute units
does nothing. The only impactful change is higher memory bandwidth (e.g., HBM) or eliminating
DRAM traffic via near-memory / in-memory computation.
