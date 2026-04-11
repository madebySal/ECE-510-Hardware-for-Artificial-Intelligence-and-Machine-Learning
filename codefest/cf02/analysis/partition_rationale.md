# HW/SW Partition Proposal — BNN Inference Accelerator

## Project: Binary Neural Network Inference Accelerator
## ECE 510 Spring 2026 — Codefest 2 CLLM

---

## (a) Which kernels to accelerate in hardware, and why the roofline supports it

The kernel to accelerate is **the XNOR-popcount linear layer** (`bnn_layer_xnor`),
which accounts for 99.3% of total BNN inference runtime (profiling data:
`codefest/cf02/profiling/project_profile.txt`). This kernel implements the binary
dot product: for each output neuron, XNOR the packed input activation bits with the
packed weight bits, then popcount the result to obtain the integer activation sum.

The roofline analysis supports this choice directly. The kernel's arithmetic intensity
is **15.11 FLOP/byte** (see `ai_calculation.md`). On the target CPU (ridge point ≈ 38
FLOP/byte), this places the kernel firmly in the memory-bound regime — it cannot
reach the compute ceiling because data movement is the bottleneck. In hardware, we
can eliminate most of that DRAM traffic: the entire weight matrix for all three layers
fits in **31 KB** of on-chip SRAM (because binary weights are packed 1 bit per value
vs. 32 bits in FP32). With on-chip SRAM at ~200 GB/s effective bandwidth, the same
15.11 FLOP/byte kernel becomes much closer to the ridge point, and the hardware
compute ceiling (XNOR-popcount gates operating at ~100+ GOPS) is reachable.

Additionally, the XNOR-popcount operation maps trivially to hardware: a single XNOR
gate per bit, followed by a popcount tree (Wallace tree or LUT-based). This is far
simpler than designing an FP32 MAC unit, making synthesis with OpenLane 2 realistic.

## (b) What the software baseline will continue to handle

The host CPU will handle everything outside the binary linear layers:
- Loading the model weights and activations from memory into the chiplet via the
  hardware interface (AXI4-Lite or SPI)
- Any pre-processing of input data (normalization, reshaping)
- The binarization step (sign function) between layers — this is a trivial comparison
  that does not warrant hardware acceleration
- The final argmax over the output layer to produce a class prediction
- Any control flow, batching, or scheduling logic

These operations are either trivially fast or inherently sequential and data-dependent,
making them poor candidates for a fixed-function accelerator.

## (c) Interface bandwidth required to avoid becoming interface-bound

Target throughput: 1,000 inferences per second (reasonable for keyword spotting).

Per inference data transfers:
- Input activation: 784 bits = **98 bytes**
- Output result: 10 × FP32 = **40 bytes**
- Weights are loaded once and cached on-chip; not transferred per inference

Required interface bandwidth:
```
BW = (98 + 40) bytes × 1,000 inferences/s = 138,000 bytes/s ≈ 0.14 MB/s
```

Even SPI at 50 Mbit/s = 6.25 MB/s is **44× the required bandwidth**. AXI4-Lite
easily sustains this. The design is not interface-bound at this operating point.

At a more aggressive 1,000,000 inferences/s:
```
BW = 138 bytes × 1,000,000 = 138 MB/s
```

This still fits within AXI4-Lite (hundreds of MB/s at 100 MHz with 32-bit data bus).
SPI would become the bottleneck at ~4.5× insufficient bandwidth, which justifies
choosing AXI4-Lite as the interface.

## (d) Compute-bound or memory-bound, and does the accelerator change that?

**On CPU (current):** The BNN kernel is **memory-bound** at AI = 15.11 FLOP/byte
vs. CPU ridge point ≈ 38 FLOP/byte. The kernel cannot reach the CPU compute ceiling
because DRAM bandwidth (68 GB/s) is exhausted first.

**On hardware accelerator (target):** The accelerator moves weights into **on-chip
SRAM** (~200 GB/s internal bandwidth). The effective AI of the XNOR-popcount
compute engine against on-chip SRAM bandwidth is:

```
Ridge point (accelerator) = 50,000 GOPS / 200 GB/s = 250 FLOP/byte
```

With AI = 15.11 FLOP/byte vs. ridge point 250 FLOP/byte, the design **remains
memory-bound even in hardware**, but now against the SRAM bandwidth rather than
DRAM. This motivates a **weight-stationary dataflow**: keep the weight matrix
resident in on-chip SRAM across multiple inferences so the dominant traffic is
only the input activations (98 bytes/inference), which dramatically reduces the
effective bytes-per-inference and shifts AI upward. With weight reuse across a
batch of 64 inferences, the effective AI rises to ~970 FLOP/byte, moving the
design into the compute-bound regime and making full utilization of the XNOR
compute array achievable.
