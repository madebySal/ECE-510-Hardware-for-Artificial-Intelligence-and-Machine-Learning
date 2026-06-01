# CMAN — Arithmetic Intensity Analysis
## BNN Accelerator Dominant Kernel, From First Principles
### ECE 510 Spring 2026 — Codefest 09

---

## Task 1 — Dominant Kernel: Dimensions and Data Type

**Kernel:** XNOR-popcount dot product — one BNN fully-connected neuron.

| Parameter | Value |
|-----------|-------|
| Kernel name | XNOR-popcount (binary dot product) |
| Input vector | N = 64 binary activations, packed as 8 bytes |
| Weight vector | N = 64 binary weights, packed as 8 bytes |
| Data type | 1-bit (binary ±1, stored as single bits) |
| Output | 1-bit neuron result (thresholded popcount) |
| Hardware | `compute_core.sv` — one invocation per clock after `data_ready` |

The design is implemented at N = 64 (simulation target). The production MNIST target is N = 784;
scaling notes are included at the end.

---

## Task 2 — FLOPs Count (One Kernel Invocation, N = 64)

Binary XNOR-popcount maps to a multiply-accumulate pattern:
- **XNOR step:** N XNOR operations, one per (activation, weight) pair → **64 ops**
- **Popcount step:** Sum all N XNOR outputs via an adder tree, which requires N − 1 = 63 additions → **63 ops**
- **Threshold compare:** 1 comparison (pop_comb > N/2) → **1 op**

```
Total FLOPs = N + (N − 1) + 1 = 2N = 2 × 64 = 128 binary ops
            = 64 XNOR-MACs × 2 = 128 FLOPs
```

**One invocation = 128 binary FLOPs.**

> Note: "Binary FLOPs" treats each 1-bit operation (XNOR, 1-bit add, compare) as 1 FLOP, consistent with
> the XNORNet / XNOR-popcount literature. These are not IEEE-754 FLOPs; the metric is stated explicitly.

---

## Task 3 — Bytes Transferred from Off-Chip Memory

The reuse pattern is **activation reuse** (not weight reuse): a single activation vector is fixed per inference
while weight rows for each of M output neurons are streamed one by one. This pattern applies to a single
fully-connected BNN layer and is analogous to GEMV with binary inputs.

**Pattern name:** Streaming activation reuse (GEMV-style, binary activations fixed, weight rows streamed).

### Lower Bound on AI — No Data Reuse (per neuron, fresh loads)

Both the activation vector and the weight row are loaded from off-chip memory for every neuron:

```
Bytes (no reuse) = activations + weights
                 = (N / 8) + (N / 8)
                 = (64 / 8) + (64 / 8)
                 = 8 + 8
                 = 16 bytes per neuron invocation
```

### Upper Bound on AI — Perfect Activation Reuse (M output neurons per layer)

Activations (8 bytes) are loaded once into the on-chip register file and reused for all M output neurons.
Weight rows are still streamed (M × 8 bytes). As M → ∞, the activation cost becomes negligible:

```
Bytes (perfect act. reuse, M neurons) = activation_load + M × weight_row
                                       = (N / 8) + M × (N / 8)
                                       = 8 + 8M   bytes

FLOPs (M neurons)                      = M × 2N = 128M

AI (M neurons) = 128M / (8 + 8M) = 128M / (8(1 + M))

As M → ∞:  AI_high → 128 / 8 = 16 binary FLOPs/byte
```

For M = 256 (MNIST Layer 1 output count):
```
AI(256) = 128 × 256 / (8 + 8 × 256) = 32,768 / 2,056 ≈ 15.93 ≈ 16 FLOPs/byte
```

---

## Task 4 — Arithmetic Intensity and Roofline

### Arithmetic Intensity Summary

| Bound | Formula | Value |
|-------|---------|-------|
| Lower (no reuse) | 2N / (2 × N/8) = 8 | **8.0 FLOPs/byte** |
| Upper (perfect act. reuse) | 2N / (N/8) = 16 | **16.0 FLOPs/byte** |

### sky130A ASIC Platform — Nominal Roofline Parameters

The target platform is sky130A / sky130_fd_sc_hd running at 100 MHz (from synthesis, nom_tt_025C_1v80).

| Parameter | Value | Source |
|-----------|-------|--------|
| Clock frequency | 100 MHz | M3 synthesis constraint |
| Useful ops per clock cycle | 128 binary ops (full XNOR-popcount in 1 cycle) | compute_core.sv |
| **Peak compute** | 128 × 100 × 10⁶ = **12.8 GOPS** | synthesis |
| **Interface bandwidth (SPI)** | 12.5 MHz × 1 bit/cycle ÷ 8 = **1.5625 MB/s** | SPI mode-0, 12.5 MHz SCK |
| On-chip register file BW | 128 bytes × 100 MHz = 12.8 GB/s | negligible bottleneck |
| **Ridge point** | 12.8 × 10⁹ / (1.5625 × 10⁶) = **8,192 FLOPs/byte** | compute / BW |

### Attainable Performance at Each AI Bound

| AI Bound | Value | Attainable Performance |
|----------|-------|------------------------|
| No reuse (lower) | 8 FLOPs/byte | 8 × 1.5625 MB/s = **12.5 MOPS** |
| Perfect reuse (upper) | 16 FLOPs/byte | 16 × 1.5625 MB/s = **25.0 MOPS** |

Both bounds fall far left of the ridge point (8,192 FLOPs/byte) → **the kernel is heavily memory-bandwidth bound via the SPI interface.**

### Roofline Sketch

See `codefest/cf09/cman_roofline_sketch.png`.

The sketch shows:
- BW ceiling line (slope = 1.5625 MB/s)
- Compute ceiling (12.8 GOPS horizontal line)
- Ridge point at (8192 FLOPs/byte, 12.8 GOPS)
- Kernel lower AI bound marked at (8, 12.5 MOPS)
- Kernel upper AI bound marked at (16, 25.0 MOPS)
- Both kernel points land on the BW ceiling slope (memory-bound region)

---

## Task 5 — Bottleneck Identification and Improvement

### Is the design limited by interface bandwidth, on-chip memory bandwidth, or compute units?

**Interface bandwidth (SPI).** The SPI interface at 12.5 MHz delivers only 1.5625 MB/s, while the
compute core can sustain 12.8 GOPS peak — a factor of **8,192× mismatch** at the ridge point.
The on-chip register file bandwidth (~12.8 GB/s) is not a bottleneck.

The measured co-simulation throughput (8.88 MOPS per neuron evaluation) confirms this: the system
achieves less than 0.07% of its peak compute capacity because 99.93% of each inference cycle is spent
transferring data through SPI rather than computing.

### Single Highest-Leverage Change

**Replace the SPI interface with a 32-bit parallel AXI4-Lite interface at 100 MHz.**

- AXI4-Lite at 100 MHz delivers 400 MB/s (32 bits/cycle × 100 MHz / 8) — a **256× bandwidth increase** over SPI.
- This shifts the ridge point from 8,192 to 32 FLOPs/byte, placing the kernel's AI bounds (8–16) near or above the new ridge.
- The critical path (currently the regfile write-enable decode tree) would also shorten because the 7-bit SPI address decode at 80 ns/bit serial timing is replaced by parallel word writes.
- The compute core itself requires no changes.

---

## Scaling Note: N = 784 Production Target

At N = 784 (full MNIST inference, 784 inputs per neuron):

| Parameter | N = 64 (implemented) | N = 784 (production) |
|-----------|---------------------|---------------------|
| FLOPs/neuron | 128 | 1,568 |
| Bytes/neuron (no reuse) | 16 | 196 |
| AI_low | 8.0 FLOPs/byte | 8.0 FLOPs/byte |
| AI_high (M=256) | 15.9 FLOPs/byte | 15.9 FLOPs/byte |
| SPI transfer time/neuron | 11.5 µs | 141 µs |

AI bounds are independent of N — the ratio of ops to bytes stays constant for this kernel.
The SPI bottleneck worsens as N increases (more bytes to transfer per neuron).
