# Roofline Analysis — BNN Accelerator
## ECE 510 Spring 2026 — Codefest 09 (CLLM Task 9)

The accelerator lands at 8.88 MOPS on the sky130A roofline, measured from co-simulation. This
point sits on the BW ceiling slope at AI = 8 FLOPs/byte — exactly where the roofline model predicts
for a memory-bandwidth-limited design with no activation reuse. The prediction from Task 4 (12.5 MOPS
at AI = 8) slightly exceeds the measured value; the gap of ~3.6 MOPS is explained by protocol
overhead: each SPI byte transfer requires a start bit (9 bits total per byte instead of 8), adding
~11% to the transfer time and reducing effective bandwidth from 1.5625 MB/s to approximately
1.42 MB/s. At 1.42 MB/s × 8 FLOPs/byte = 11.4 MOPS, the model matches the measurement closely.

The larger gap between measured (8.88 MOPS) and the CPU baseline (4,100 MOPS) is not a matter
of arithmetic intensity — both kernels share AI ≈ 8 FLOPs/byte — but entirely a bandwidth gap:
the CPU's DDR5 at ~50 GB/s outpaces SPI by a factor of 32,000×. The accelerator's peak compute
(12.8 GOPS) exceeds the CPU's (4.1 GOPS), but this advantage is inaccessible while the SPI
interface under-feeds the compute core. Replacing SPI with a 32-bit AXI4 bus at 100 MHz would
raise effective bandwidth to ~400 MB/s, shift the ridge point to 32 FLOPs/byte, and allow the
accelerator to approach its compute ceiling — closing the throughput gap with the CPU while
retaining the 5,400× power advantage.
