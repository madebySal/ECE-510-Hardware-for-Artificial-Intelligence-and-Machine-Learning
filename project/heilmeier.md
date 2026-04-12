# Heilmeier Catechism – BNN Inference Accelerator
## ECE 510 Spring 2026 | Saleh Esmaeil

---

## Q1. What are you trying to do?

Build a custom hardware chiplet that accelerates inference of Binary Neural Networks (BNNs) —
neural networks where weights and activations are constrained to +1/−1. The target architecture
is a fully-connected BNN [784→256→128→10] for keyword spotting or gesture recognition on
edge MCU-class devices. The core operation replaces FP32 multiply-accumulate with XNOR-popcount:
each multiplication reduces to a single XNOR gate and accumulation reduces to popcount, enabling
fast, low-power inference without a GPU or general-purpose processor.

---

## Q2. How is it done today, and what are the limits?

Today, BNN inference runs on CPU using FP32 NumPy matrix multiplies, even though the weights
are binary (+1/−1). Profiling the [784→256→128→10] BNN on an Intel i7-1165G7 shows:

- **Forward pass time:** 149 µs per inference (6,692 samples/sec)
- **Dominant kernel:** Layer 1 matmul `[1×784] @ [784×256]` via `numpy.matmul`
- **Arithmetic intensity:** 0.50 FLOP/byte (FP32 weights, DRAM no-reuse)
- **Roofline result:** Memory-bound — attainable 20 GFLOP/s vs 150 GFLOP/s compute ceiling

The fundamental limit is that FP32 weights generate 802,816 bytes of DRAM traffic per inference
for W1 alone, despite each weight carrying only 1 bit of information. The software is wasting
31 out of every 32 bits fetched from memory. This is a structural inefficiency that cannot be
fixed by faster CPUs or better software — it requires a hardware change to the memory layout
and compute primitive.

---

## Q3. What is your approach and why is it better?

This project implements a custom XNOR-popcount compute engine in SystemVerilog for BNN
inference. The key hardware changes are:

1. **1-bit weight packing:** W1 is stored as 784×256 bits = 25,088 bytes (vs 802,816 bytes FP32),
   a 32× reduction in weight memory traffic.
2. **XNOR-popcount compute:** Each FP32 multiplier is replaced by a single XNOR gate; accumulation
   uses a popcount tree. This enables massive parallelism at <1% of FP32 silicon area.
3. **On-chip weight caching:** All packed weights (≈31 KB total) fit in on-chip SRAM, eliminating
   repeated DRAM access after startup.

**Why it is better — roofline argument:**  
With 1-bit packed weights, arithmetic intensity rises from 0.50 to ≈16 XNOR-ops/byte (Layer 1).
The design target is 1 TOPS XNOR compute at 200 GB/s on-chip SRAM bandwidth (ridge = 1.0).
At AI = 16, the kernel is compute-bound on the accelerator — the HW design point sits 16× above
the ridge, achieving full compute utilization. The projected speedup over the software baseline
is 10–50× at a fraction of the energy cost, making the design viable for battery-powered edge
inference.

**References:**  
- Rastegari et al., "XNOR-Net," ECCV 2016  
- Courbariaux et al., "BinaryConnect," NeurIPS 2015  
- Umuroglu et al., "FINN," FPGA 2017
