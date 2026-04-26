# M2 Precision Documentation — BNN Inference Accelerator

## Chosen Precision: 1-bit Binary (±1)

Both weights and activations are binarized to {+1, −1}, represented in hardware as a single bit (1 = +1, 0 = −1).

---

## Why 1-bit?

| Metric | FP32 BNN baseline | 1-bit hardware |
|--------|-------------------|----------------|
| Weight storage (784×256) | 800 KB | 25 KB |
| MAC operation | FP multiply-add | XNOR + popcount |
| Gate count (1 neuron, N=784) | ~50k gates (est.) | ~800 gates (XNOR array + adder tree) |
| Throughput | 1 cycle/multiply | 1 cycle for all 784 operations (parallel) |

The BNN algorithm *defines* weights as ±1 — there is no higher-precision version to round down from. Binary precision is the algorithm, not an approximation of it.

---

## Quantization Error Analysis

### What "quantization error" means here

In a traditional INT8 accelerator, quantization error is the difference between the FP32 weight and its rounded integer representation. For a BNN, the concept is different: the quantization happens at **training time** (sign of the real-valued weight), not in hardware. The hardware implements the already-binarized model exactly.

**Hardware-induced error: zero.**  
The XNOR+popcount operation is lossless — no rounding, no truncation beyond what the binarized weights already encode.

### Accuracy gap vs FP32 (inherent to BNN algorithm)

On MNIST (the target task from the Heilmeier proposal):

| Model | Test accuracy |
|-------|--------------|
| FP32 MLP (784→256→10) | ~98.2% |
| BNN (same topology, ±1 weights/acts) | ~96–97% |
| **Accuracy gap** | **~1–2 pp** |

This gap is the cost of the binarization approximation. It is a property of the BNN *algorithm*, not of this hardware implementation. The hardware reproduces the binarized model bit-exactly.

### Popcount accumulator width

With N = 784 inputs, the popcount result lies in [0, 784].  
Required bits: ⌈log₂(784 + 1)⌉ = 10 bits.  
`bnn_layer.sv` uses `$clog2(N)+1 = 11 bits` — one extra bit of headroom, no overflow possible.

### Threshold

A neuron fires (+1) when more than half of the XNOR bits are 1, i.e., `pop > N/2 = 392`.  
This is the binary sign function: sign(W·x) where W·x is approximated by `2·pop − N`.

---

## No Dequantization Step

Traditional quantized networks require dequantization (multiply by scale S) before passing results to the next layer. Binary networks do not: the output of each layer is again a 1-bit value (sign of popcount), so the scale cancels and no floating-point multiply is needed anywhere in the inference chain.

---

## SPI Interface Bandwidth vs Precision

Loading one weight row (784 bits) over 6.25 MB/s SPI:

- Bytes needed: ⌈784/8⌉ = 98 bytes
- Transfer time at 6.25 MB/s: 98 / 6.25×10⁶ ≈ 15.7 µs
- Inference target: 10,000/sec → 100 µs budget per inference
- **Headroom: 6.4×** (weight load uses only 15.7% of the time budget)

At INT8 precision, the same row would require 784 bytes → 125.4 µs, exceeding the 100 µs budget. Binary precision is therefore **necessary** to meet the throughput requirement over the chosen SPI interface.
