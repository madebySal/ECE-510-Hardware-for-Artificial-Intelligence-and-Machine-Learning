# Project Scope Assessment — BNN Inference Accelerator

**Updated:** 2026-05-17 (post CF07 synthesis)

## Current Scope

The project implements a single-neuron BNN inference core: 64-bit (sim) / 784-bit
(production) XNOR-popcount compute core connected to a host MCU via SPI slave interface.
The compute core produces one binary neuron output per inference trigger.

## Synthesis Result (CF07)

- **N=64 synthesis:** 2874.01 µm², 259 cells, critical path 4.499 ns, slack +5.501 ns
  at 10 ns clock. Timing is comfortably met.
- **Dominant logic:** XOR/XNOR reduction tree for popcount (56% of cells). The design
  is clean: two DFFs, no memory, no combinational loops.

## Scope Assessment

**Scope is confirmed as-is for M3.** The N=64 core synthesizes cleanly and meets timing
with margin. The N=784 production configuration will be approximately 12× larger in
area but is expected to still meet 10 ns timing (popcount tree grows from ~7 to ~10
levels, adding ~2 ns to the critical path).

The SPI interface module was not synthesized in CF07; it should be included for the
M3 full-chip synthesis. The interface is register-file-based (128 × 8-bit regs) and
will likely dominate area in the full N=784 design due to the large activation and
weight register banks (2 × 98 bytes = 196 flip-flop bytes = 1568 DFFs).

**Potential risk:** At N=784, the register file (1568 DFFs) will be ~30–40× larger
in area than the compute core alone. If die area is constrained, reducing N or
switching to SRAM-based weight storage may be necessary — to be evaluated at M3.
