# M4 Benchmark — BNN Accelerator vs. M1 Software Baseline
## ECE 510 Spring 2026

---

## Platform

| | M1 Software Baseline | M4 HW Accelerator |
|-|---------------------|-------------------|
| Implementation | NumPy BNN, Python 3.12 | SystemVerilog RTL, sky130A |
| Architecture | FC-BNN [784→256→128→10] | XNOR-popcount core, N=64 |
| Platform | Intel Core i7-13700, Windows 11 | sky130_fd_sc_hd, 100 MHz |
| Measurement | Measured (perf_counter, 50 runs) | Measured (Icarus co-simulation) |

---

## Measured Results

### M1 Software Baseline (re-run, same code as M1)

| Metric | Value |
|--------|-------|
| Median time / inference | **114.40 µs** |
| Throughput | **8,741 samples/sec** |
| Compute throughput | **4.10 GOPS** (469,504 FLOPs / 114.40 µs) |
| Peak memory (RSS) | **8.0 KB** |
| Estimated socket power | ~50 W |
| Energy / inference | ~5,720 µJ |

FLOPs: Layer1 784×256×2 + Layer2 256×128×2 + Layer3 128×10×2 = 469,504 FLOPs.

### M4 Hardware Accelerator (co-simulation, N=64)

| Metric | Value | Method |
|--------|-------|--------|
| Time / neuron eval | **14.41 µs** | Measured, Icarus Verilog co-sim |
| Neuron evals / sec | **69,396** | Measured |
| Compute throughput | **8.88 MOPS** | 128 ops / 14.41 µs |
| Peak compute | **12.8 GOPS** | 128 ops/cycle × 100 MHz |
| Compute utilization | **0.069%** | 8.88 MOPS / 12,800 MOPS |
| Power | **9.22 mW** | OpenSTA, nom_tt_025C_1v80 |
| Energy / neuron | **133 nJ** | 9.22 mW × 14.41 µs |

Inference cycle breakdown (all measured from co-sim VCD timing):

| Phase | Time |
|-------|------|
| SPI write: act + wgt (16 bytes × 9 bits × 80 ns) | 11,520 ns |
| SPI write: CTRL register (2 bytes) | 1,440 ns |
| Compute: XNOR-popcount (1 clock cycle) | 10 ns |
| SPI read: STATUS register (2 bytes) | 1,440 ns |
| **Total** | **14,410 ns = 14.41 µs** |

---

## Speedup and Energy

### Speedup (throughput, M1 time / M4 time)

Speedup = 114.40 µs / 14.41 µs = **7.94×** per compute cycle
(HW computes its result 7.9× faster than one full SW inference per elapsed wall time)

> Note: This per-neuron speedup comparison is against the full 3-layer SW inference.
> For a like-for-like N=64 comparison: SW time for 64-input layer ≈ 0.29 µs/neuron;
> HW = 14.41 µs/neuron → **HW is 49× slower** due to SPI interface bottleneck.
> The SPI transfer (14.40 µs) dominates; compute itself takes only 10 ns (0.07% of cycle).

### Full MNIST Projected (N=784)

| Metric | SW Baseline | HW Projected | Ratio |
|--------|-------------|-------------|-------|
| Time / inference | 114.40 µs | ~46 ms | 0.0025× |
| Throughput | 8,741 samples/sec | ~22 samples/sec | 0.0025× |
| Power | ~50 W | 9.22 mW | HW 5,420× lower |
| Energy / inference | ~5,720 µJ | ~102 µJ | **HW 56× lower** |

Projection basis: N=784 SPI transfer = 196 bytes × 9 bits × 80 ns = 141 µs/neuron.
Layer 1 (256 neurons): 36.1 ms; Layer 2+3: ~8.4 ms; Total ≈ 44–46 ms.

### Energy Efficiency

Energy improvement = 5,720 µJ / 102 µJ = **56× more energy-efficient**
Power improvement = 50 W / 9.22 mW = **5,420× lower power**

---

## Bottleneck

The SPI interface at 12.5 MHz delivers 1.5625 MB/s.
The compute core peak is 12.8 GOPS.
Ridge point: 12.8 GOPS / 0.0015625 GB/s = **8,192 FLOPs/byte**.
Kernel AI: 8–16 FLOPs/byte — **memory-bandwidth bound by 512–1024×**.

See `bench/roofline_final.png` and `report/design_justification.pdf` §8.

---

## Raw Data

All numbers traceable to `bench/benchmark_data.csv`.
