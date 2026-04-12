# HW/SW Partition Proposal – BNN Inference Accelerator
## ECE 510 – CF02 | Spring 2026

## (a) Kernel to Accelerate in Hardware

The kernel selected for hardware acceleration is the binary matrix multiply — the XNOR-popcount
operation that implements each fully-connected layer of the BNN ([784→256], [256→128], [128→10]).
The roofline analysis directly motivates this choice: the software baseline achieves an arithmetic
intensity of only 0.50 FLOP/byte, placing the kernel deep in the memory-bound region of the roofline
(ridge point = 3.75 FLOP/byte for the i7-1165G7). Attainable performance is capped at 20 GFLOP/s
against a 150 GFLOP/s compute ceiling. The gap exists because FP32 weights dominate memory
traffic. A hardware accelerator using 1-bit packed binary weights reduces the weight footprint by 32×:
W1 drops from 802,816 bytes (FP32) to 25,088 bytes (1-bit), shifting arithmetic intensity to
401,408 / 25,088 ≈ **16 XNOR-ops/byte** — well into the compute-bound regime of the proposed
on-chip design (ridge point = 200 GOPS / 200 GB/s = 1.0). The XNOR-popcount operation is also
extremely area- and energy-efficient in silicon: each XNOR gate replaces a full FP32 multiplier,
enabling massive parallelism at a fraction of the power.

## (b) What Software Continues to Handle

The host MCU software baseline retains: (1) input preprocessing and binarization of the input
activation before transmission; (2) softmax and argmax on the 10-element output vector for
classification; (3) application-level control flow, thresholding, and result reporting; (4) model
weight loading from flash into the chiplet's on-chip SRAM at startup. These operations account
for a negligible fraction of total runtime and involve no repeated matrix operations, so they do
not benefit from the XNOR-popcount hardware.

## (c) Interface Bandwidth Requirement

**Target operating point:** 10,000 inferences/sec (10× the software baseline throughput of ~6,700/sec).

**Required data per inference:**
- Input: 784 bits = 98 bytes (1-bit packed binary input)
- Output: 10 × 4 bytes = 40 bytes (FP32 logits)
- Weights are cached on-chip after startup (loaded once)

**Required interface bandwidth:**
```
Bandwidth = (98 + 40) bytes × 10,000 inferences/sec = 1,380,000 bytes/sec ≈ 1.38 MB/s
```

**Chosen interface:** SPI at 50 Mbit/s = 6.25 MB/s rated bandwidth.  
**Conclusion:** Required bandwidth (1.38 MB/s) is well below SPI's 6.25 MB/s rated bandwidth.  
The design is **not interface-bound** at this operating point. SPI provides a 4.5× headroom margin.

## (d) Bound Classification and Expected Change

On the current CPU, the BNN matmul is **memory-bound** (AI = 0.50 FLOP/byte, below ridge at 3.75).
The hardware accelerator is designed to move the kernel into the **compute-bound** regime: with 1-bit
packed weights (AI ≈ 16 XNOR-ops/byte) and a target ridge point of 1.0, the kernel sits 16× above
the ridge, meaning compute throughput — not memory bandwidth — becomes the binding constraint.
This is the desired outcome: it allows the hardware to exploit the full 1 TOPS XNOR throughput of
the compute engine, delivering the speedup that justifies the chiplet design.
