# Heilmeier Catechism — BNN Inference Accelerator
## ECE 510 Spring 2026 — Updated after profiling (Codefest 2)

---

## Q1. What are you trying to do?

Design and synthesize a custom co-processor chiplet that accelerates Binary Neural
Network (BNN) inference by replacing floating-point multiply-accumulate (MAC)
operations with XNOR-popcount operations in hardware. The target kernel is the
binary linear layer: for each output neuron, XNOR the 784-bit packed input
activation vector with the 784-bit packed weight vector, then popcount the result
to obtain the integer activation sum. This operation is provably reducible to a
single XNOR gate per bit and a popcount tree — no multipliers required. The chiplet
will be described in SystemVerilog, synthesized using OpenLane 2, and connected to
a host via an AXI4-Lite interface.

---

## Q2. How is it done today, and what are the limits?

BNN inference today runs on general-purpose CPUs or GPUs using INT8 or FP32
arithmetic units, even though the actual math only requires 1-bit XNOR and
integer popcount. Profiling a pure-Python/NumPy BNN forward pass on a CPU
(architecture [784→256→128→10], batch size 1) shows:

- **Float32 path** (using NumPy matrix multiply): median **0.060 ms** per inference
- **XNOR-popcount path** (Python-emulated, bit-packed): median **6.651 ms** per inference

The Python emulation is slower than FP32 because Python-level loops emulate what
would be a single hardware instruction (XNOR + popcount on 64-bit words). The
dominant bottleneck is `bnn_layer_xnor`, accounting for **99.3% of XNOR-path runtime**.

The arithmetic intensity of the XNOR-popcount kernel is **15.11 FLOP/byte**
(full network, DRAM, no reuse). On a CPU with ridge point ≈ 38 FLOP/byte, this
places the kernel in the **memory-bound regime** — the kernel cannot reach the
compute ceiling because DRAM bandwidth is exhausted first. The key limitation is
that no current general-purpose hardware provides native XNOR-popcount units wide
enough to process 784-bit vectors in a single cycle, forcing software to decompose
the operation into scalar or 64-bit chunks across many clock cycles.

---

## Q3. What is new in your approach, and why do you think it will be successful?

The accelerator implements a dedicated **XNOR-popcount compute array** where each
processing element performs: `output[i] = popcount(XNOR(activation_packed, weight_packed[i]))`.
With binary weights packed to 1 bit/value, the entire weight matrix for a
[784→256→128→10] BNN occupies only **~31 KB** of on-chip SRAM — small enough to
keep resident between inferences, eliminating per-inference DRAM traffic for weights.

This approach is grounded in the roofline analysis:
- The kernel AI of 15.11 FLOP/byte is memory-bound on CPU because DRAM bandwidth
  (≈68 GB/s) limits throughput.
- Moving weights on-chip (SRAM bandwidth ≈200 GB/s) with weight-stationary dataflow
  and a batch of 64 inferences raises effective AI to **~970 FLOP/byte** — well into
  the compute-bound regime for the target accelerator.
- The hardware primitives (XNOR gate, Wallace tree popcount) are synthesizable and
  well-characterized; no novel circuit design is required, making OpenLane 2 synthesis
  realistic within the project timeline.

The approach is expected to succeed because: (1) the math is simple and maps directly
to digital logic; (2) the weight memory fits on-chip at a reasonable area cost;
(3) published work (XNOR-Net, FINN) confirms 10–100× energy efficiency gains over
FP32 equivalents in silicon. A measured speedup over the CPU FP32 baseline
(0.060 ms/inference) is achievable even at modest clock frequencies.

---

## References
- Rastegari et al., "XNOR-Net," ECCV 2016
- Courbariaux et al., "BinaryConnect," NeurIPS 2015
- Umuroglu et al., "FINN: Fast, Scalable BNN Inference," FPGA 2017
