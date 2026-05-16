# CMAN Sparsity Analysis
**ECE 410/510 — Codefest 07**  
**N = 512, Sparsity = s (fraction of zeros)**
 
---
 
## Task 1 — Four Expressions for Dense and Sparse Compute/Memory
 
### (a) Dense MVM Compute (FLOPs)
 
Each of the N² elements requires 1 multiply and 1 add = 2 FLOPs:
 
```
Dense FLOPs = 2 × N² = 2 × 512² = 524,288 FLOPs
```
 
### (b) Dense Memory Bytes
 
Each element is FP32 = 4 bytes:
 
```
Dense Memory = 4 × N² = 4 × 512² = 1,048,576 bytes = 1 MB
```
 
### (c) Sparse Compute (FLOPs, as a function of s)
 
Only non-zero weights are computed. Fraction of non-zeros = (1 - s):
 
```
Sparse FLOPs = 2 × N² × (1 - s)
```
 
### (d) Sparse Memory Bytes (CSR, as a function of s)
 
CSR stores three arrays:
- **Values array**: one FP32 per non-zero → 4 bytes each
- **Column index array**: one INT32 per non-zero → 4 bytes each
- **Row pointer array**: N+1 entries → 4 bytes each
```
Sparse Memory = N²(1-s) × 8 + 4(N+1) bytes
             = 512²(1-s) × 8 + 4(513) bytes
```
 
---
 
## Task 2 — FLOPs Speedup and s for 2× Speedup
 
FLOPs speedup = Dense FLOPs / Sparse FLOPs:
 
```
Speedup = 2N² / 2N²(1-s) = 1 / (1-s)
```
 
For 2× speedup:
 
```
1 / (1-s) = 2
1 - s = 0.5
s = 0.5
```
 
**Answer: s = 0.5 (50% sparsity) gives 2× FLOPs speedup.**
 
---
 
## Task 3 — Memory Breakeven Sparsity
 
Set sparse memory = dense memory and solve for s:
 
```
8N²(1-s) + 4(N+1) = 4N²
```
 
For large N, drop the small 4(N+1) term:
 
```
8N²(1-s) = 4N²
8(1-s) = 4
1-s = 0.5
s = 0.5
```
 
**Answer: Breakeven at s = 0.5 (50% sparsity).**  
Above this sparsity level, CSR format uses less memory than dense storage.
 
---
 
## Task 4 — End-to-End Speedup at s=0.9 (Memory-Bound, 320 GB/s)
 
**Dense memory bytes:**
```
4 × N² = 4 × 512² = 1,048,576 bytes
```
 
**Sparse memory bytes at s = 0.9:**
```
8 × N²× (1-s) = 8 × 512² × 0.1 = 209,715 bytes
```
 
**Dense execution time:**
```
1,048,576 / (320 × 10⁹) = 3.28 µs
```
 
**Sparse execution time:**
```
209,715 / (320 × 10⁹) = 0.655 µs
```
 
**End-to-end speedup:**
```
Speedup = 3.28 / 0.655 ≈ 5×
```
 
**Answer: At s=0.9 with 320 GB/s memory bandwidth, sparse achieves ~5× speedup over dense.**
 
