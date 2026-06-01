# CMAN — Arithmetic Intensity & Roofline Analysis
**ECE 410/510 — Codefest 09**  
**Project: BNN Inference Accelerator — Saleh Esmaeil**

---

## Task 1 — Dominant Kernel

| Property | Value |
|---|---|
| Kernel | Matrix-Vector Multiply (MVM) |
| Dimensions | [1×784] @ [784×256] |
| Data type | 1-bit binary (±1, XNOR+popcount) |
| Operation | 784 XNOR + popcount per output neuron × 256 neurons |
| Operating point | Single inference, batch size = 1 |

---

## Task 2 — FLOPs Count

Each output neuron requires 784 XNOR operations + 1 popcount accumulation.  
Counting each XNOR and each accumulation as 1 FLOP each → 2 FLOPs per weight:

```
FLOPs = 2 × 784 × 256 = 401,408 FLOPs
```

---

## Task 3 — Bytes Transferred (Two Bounds)

### Reuse Pattern
This is a **weight-stationary** dot product kernel.  
Weights can be loaded once and reused across the batch.  
For batch=1, the two bounds are:

### Lower Bound AI — No Data Reuse

Load everything from off-chip memory every inference:

| Data | Size |
|---|---|
| Weights: 784×256 bits | 784×256/8 = **25,088 bytes** |
| Input activations: 784 bits | 784/8 = **98 bytes** |
| Output: 256 bits | 256/8 = **32 bytes** |

```
Total bytes (no reuse) = 25,088 + 98 + 32 = 25,218 bytes

AI_lower = 401,408 / 25,218 = 15.92 FLOP/byte
```

### Upper Bound AI — Perfect On-Chip Weight Reuse

Weights fully cached in on-chip SRAM. Only activations transferred:

```
Total bytes (full reuse) = 98 + 32 = 130 bytes

AI_upper = 401,408 / 130 = 3,087.75 FLOP/byte
```

---

## Task 4 — Roofline Analysis

### Platform Specs (sky130 PDK, 100 MHz ASIC target)

| Parameter | Value |
|---|---|
| Peak compute | 12 GOPS |
| On-chip SRAM BW | ~10 GB/s |
| ASIC ridge point | 12 / 10 = **1.2 FLOP/byte** |
| SPI interface BW | 6.25 MB/s = 0.00000625 GB/s |

### SW Baseline (i7-13700, DDR5)

| Parameter | Value |
|---|---|
| Peak compute | ~400 GOPS |
| Memory BW | ~89.6 GB/s |
| CPU ridge point | 400 / 89.6 = **4.46 FLOP/byte** |

### Roofline Sketch

See: `codefest/cf09/cman_roofline_sketch.pdf`

**Key observations:**
- AI_lower = 15.92 F/B → **above ASIC ridge point (1.2 F/B)** → compute-bound on SRAM
- AI_upper = 3,088 F/B → **deep in compute-bound regime**
- Both bounds sit well above the ridge point → kernel is **compute-bound** when weights are on-chip
- However at AI_lower, the **SPI interface (6.25 MB/s)** delivers only ~99 MOPS — far below 12 GOPS peak
- This means the system is **SPI interface-bound** when weights must come from the host

---

## Task 5 — Bottleneck Identification & Improvement

### Current Bottleneck

**SPI interface bandwidth is the binding constraint.**

When weights are loaded from the host over SPI (6.25 MB/s), the effective  
throughput is limited to ~99 MOPS — roughly **120× below the 12 GOPS compute ceiling**.

Once weights are cached on-chip SRAM, the kernel becomes compute-bound  
at AI = 15.92 F/B, which is well above the ridge point and close to peak.

### Single Highest-Leverage Improvement

**Cache all weights on-chip and never reload them between inferences.**

The 25 KB weight footprint for Layer-1 fits entirely in on-chip SRAM (sky130  
supports this comfortably). Preloading weights once at startup and reusing them  
across all inferences eliminates the SPI bandwidth bottleneck entirely —  
shifting the operating point from 99 MOPS (SPI-bound) to ~12 GOPS (compute-bound).  
This is a **~120× theoretical improvement** from a single architectural decision.
