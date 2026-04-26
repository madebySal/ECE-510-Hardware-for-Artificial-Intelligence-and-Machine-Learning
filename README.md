# ECE-510-Hardware-for-Artificial-Intelligence-and-Machine-Learning
Course repository for ECE 510 – Hardware for AI and Machine Learning at Portland State University.

---

## Project: BNN Inference Accelerator

### What the module does
`project/hdl/bnn_layer.sv` implements one fully-connected BNN layer as a hardware compute core.
The dominant operation — `[1×N] @ [N×1]` with binary (±1) weights — is mapped to
**XNOR + popcount**: XNOR each activation bit with the corresponding weight bit, count the
matches, and threshold at N/2 to produce a binary output. This replaces a floating-point
matrix multiply with a single-cycle XNOR and a binary adder tree.

### Interface
**Selected interface: SPI** (see `project/m1/interface_selection.md` for full justification).

The host (ARM Cortex-M MCU) loads a 784-bit weight row once at startup over SPI
(~5 ms one-time load), then streams 98-byte packed binary activations per inference.
Required bandwidth at 10,000 inferences/sec is 1.38 MB/s — SPI at 50 Mbit/s provides
6.25 MB/s, giving 4.5× headroom. The interface is not the bottleneck.

### Precision choice
**1-bit (binary) weights and activations.** Arithmetic intensity justification:

- Software BNN (M1 baseline): 149 µs/inference, dominant kernel is `[1×784] @ [784×256]` GEMM.
- Binary GEMM replaces FP32 multiply-accumulate with XNOR+popcount, requiring no multipliers.
- Memory per inference: 98 bytes input + 31 KB weights (loaded once) → near-zero data movement
  per inference after weight caching, making the design deeply compute-bound.
- Quantization error analysis: BNN uses sign(x) binarization; no dequantization step needed.
  Accuracy loss vs FP32 baseline is inherent to the BNN architecture (not hardware-induced).
