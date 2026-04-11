# Arithmetic Intensity — BNN XNOR-Popcount Kernel

## Project: Binary Neural Network Inference Accelerator
## Dominant kernel: `bnn_layer_xnor` — XNOR-popcount linear layer

---

## Dominant kernel identification

From cProfile (`codefest/cf02/profiling/project_profile.txt`):

> The dominant kernel is `bnn_layer_xnor`, accounting for **99.3% of total runtime**
> on the XNOR-popcount path (0.135 s out of 0.136 s across 10 forward passes).

The dominant operation within that kernel is computing, for each output neuron:
1. XNOR between the packed input activation bits and the packed weight bits
2. Popcount of the XNOR result to produce the integer dot product

This is exactly the operation the hardware accelerator will implement.

---

## FLOPs derivation — Layer 1 (784 → 256), most MAC-intensive layer

For one output neuron with N_in = 784 binary inputs:
- **N_in XNOR operations** (one per input bit)
- **N_in − 1 additions** in the popcount tree ≈ N_in additions

Total per neuron ≈ 2 × N_in operations

For all N_out = 256 output neurons:

```
FLOPs = 2 × N_in × N_out
FLOPs = 2 × 784 × 256
FLOPs = 401,408
```

---

## Bytes transferred — Layer 1 (DRAM, no reuse assumed)

Binary weights are packed 1 bit per value. Inputs are also binary-packed.
Output is written as FP32 (before binarization for the next layer).

| Operand | Formula | Value |
|---|---|---|
| Weight matrix (packed) | 784 × 256 / 8 | 25,088 bytes |
| Input activation (packed) | 784 / 8 | 98 bytes |
| Output (FP32, pre-sign) | 256 × 4 | 1,024 bytes |
| **Total** | | **26,210 bytes** |

---

## Arithmetic intensity — Layer 1

```
AI = FLOPs / Bytes
AI = 401,408 / 26,210
AI = 15.32 FLOP/byte
```

---

## Full network arithmetic intensity

Network: [784 → 256 → 128 → 10]

| Layer | N_in | N_out | FLOPs | Bytes (weights + in + out) |
|---|---|---|---|---|
| Layer 1 | 784 | 256 | 401,408 | 25,088 + 98 + 1,024 = 26,210 |
| Layer 2 | 256 | 128 | 65,536 | 4,096 + 32 + 512 = 4,640 |
| Layer 3 | 128 | 10 | 2,560 | 160 + 16 + 40 = 216 |
| **Total** | | | **469,504** | **31,066** |

```
Network AI = 469,504 / 31,066 = 15.11 FLOP/byte
```

---

## Context: comparison to FP32 equivalent

For reference, a standard FP32 MLP with the same topology has:
- Weights stored as FP32 (4 bytes each): 784×256×4 = 802,816 bytes for Layer 1 alone
- AI ≈ (2 × 784 × 256) / (802,816 + 98×4 + 256×4) ≈ 0.50 FLOP/byte

The BNN XNOR-popcount kernel achieves **≈30× higher arithmetic intensity** than
FP32 for the same logical operation, because weight packing reduces bytes by 32×
while FLOPs remain the same. This is the key roofline argument for hardware acceleration.

---

## Roofline position summary

On a laptop CPU (Apple M1 equivalent: ~2.6 TFLOPS FP32, ~68 GB/s DRAM bandwidth):
- Ridge point: 2,600 / 68 ≈ 38.2 FLOP/byte
- BNN kernel AI = 15.11 FLOP/byte → **memory-bound on CPU**

On the target hardware accelerator (on-chip SRAM bandwidth ~200 GB/s, compute ~50 GOPS binary):
- Ridge point: 50,000 / 200 ≈ 250 FLOP/byte (XNOR ops vs SRAM BW)
- BNN kernel AI = 15.11 FLOP/byte → **still memory-bound**, but bottleneck is now
  on-chip SRAM at 200 GB/s rather than DRAM — a much higher bandwidth tier.
  
This motivates weight-stationary dataflow: keep weights in on-chip SRAM, stream activations.
