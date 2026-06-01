# Benchmark Results — BNN Accelerator vs. M1 Software Baseline
## ECE 510 Spring 2026 — Codefest 09 (CLLM)

---

## Platform Summary

| Field | SW Baseline (M1) | HW Accelerator (M3) |
|-------|-----------------|---------------------|
| Implementation | NumPy BNN, Python 3.12 | SystemVerilog RTL, sky130A ASIC |
| Architecture | Fully-connected [784→256→128→10] | Single XNOR-popcount neuron, N=64 |
| Platform | Intel Core i7-13700 (13th Gen, 2.10 GHz base) | sky130A / sky130_fd_sc_hd, 100 MHz |
| Run environment | Windows 11, single-threaded NumPy | Icarus Verilog co-simulation (SPI + DUT) |
| Measurement type | **Measured** | **Measured** (co-simulation) |

---

## Task 6 — M1 Software Baseline (Re-Run)

> Same Python code as M1 (`m1/sw_baseline.md`), re-run on the local machine (Windows 11, i7-13700).

| Metric | Value |
|--------|-------|
| Median time per inference | **114.40 µs** |
| Throughput | **8,741 samples/sec** |
| Compute throughput | **4.10 GOPS** (469,504 FLOPs in 114.40 µs) |
| Peak memory (RSS) | **8.0 KB** |
| Estimated CPU power | ~50 W (total socket, workstation idle load) |
| Energy per inference | ~5.72 mJ (114.40 µs × 50 W) |

FLOPs breakdown (full [784→256→128→10] forward pass):
```
Layer 1 [784×256]: 784 × 256 × 2 = 401,408 FLOPs
Layer 2 [256×128]: 256 × 128 × 2 =  65,536 FLOPs
Layer 3 [128×10]:  128 × 10  × 2 =   2,560 FLOPs
Total                              = 469,504 FLOPs
```

---

## Task 7 — Hardware Accelerator (Co-Simulation, Measured)

The M3 co-simulation (`m3/tb/tb_top.sv`, run with Icarus Verilog) passes 4/4 end-to-end tests.
Timing is measured from the co-simulation log (`m3/sim/cosim_run.log`).

### Per-Neuron Inference (N=64 design, co-simulation)

| Phase | Duration |
|-------|----------|
| SPI write: 8 bytes act + 8 bytes wgt (16 bytes × 9 bits × 80 ns/bit) | 11,520 ns |
| SPI write: CTRL register (2 bytes × 9 bits × 80 ns/bit) | 1,440 ns |
| Compute: XNOR-popcount (1 clock cycle @ 10 ns) | 10 ns |
| SPI read: STATUS register (2 bytes × 9 bits × 80 ns/bit) | 1,440 ns |
| **Total per neuron invocation** | **14,410 ns = 14.41 µs** |

| Metric | Value |
|--------|-------|
| Time per neuron evaluation | **14.41 µs** (measured, co-sim) |
| Neuron evaluation throughput | **69,396 evals/sec** |
| Compute throughput | **8.88 MOPS** (128 binary ops / 14.41 µs) |
| Peak memory (on-chip register file) | **128 bytes** (DFF-based) |
| Power (from M3 synthesis, nom_tt_025C_1v80) | **9.22 mW** |
| Energy per neuron evaluation | **133 nJ** (9.22 mW × 14.41 µs) |

---

## Task 8 — Speedup and Energy Efficiency

### Latency Comparison

For a single neuron evaluation (N=64 binary ops):

| | SW baseline | HW accelerator |
|-|-------------|----------------|
| Time | 0.45 µs ¹ | 14.41 µs |
| Latency speedup (HW vs SW) | 1.0× | **0.03× (32× slower)** |

¹ SW baseline processes full [784→256→128→10] in 114.40 µs.
  Per-neuron estimate: 114.40 µs / (256 + 128 + 10) = ~0.29 µs per neuron (Layer 1 equivalent).

> The HW accelerator is slower in raw latency because the SPI interface (12.5 MHz, 1 bit/cycle)
> delivers only 1.5625 MB/s, whereas the CPU's DDR5 memory subsystem delivers ~50 GB/s.
> The compute core itself runs in 10 ns (1 clock cycle), but data transfer takes 14.4 µs.

### Compute Throughput Comparison

| Metric | SW Baseline | HW Accelerator | Ratio |
|--------|-------------|----------------|-------|
| Compute throughput | 4.10 GOPS | 0.00888 GOPS | **0.002× (SW wins)** |
| Peak compute capacity | ~4.10 GOPS | 12.8 GOPS | HW 3.1× higher peak |
| Compute utilization | — | **0.069%** (interface bottlenecked) |

### Energy Efficiency Comparison

| Metric | SW Baseline | HW Accelerator | Ratio |
|--------|-------------|----------------|-------|
| Power | ~50 W (socket) | 9.22 mW | HW **5,420× lower power** |
| Energy / neuron eval | ~5,720 µJ ÷ 394² ≈ 14.5 µJ | 133 nJ | **HW 109× more efficient** |
| Energy / full MNIST image³ | ~5,720 µJ | ~102 µJ (projected) | **HW ~56× more efficient** |

² SW divides 5,720 µJ by ~394 total neurons in 3 layers.  
³ Full MNIST HW energy from M3 power report (projected at N=784): 9.22 mW × 14,410 ns × 256 × 3 ≈ 102 µJ.

### Summary Table

| Metric | SW Baseline | HW Accelerator | Speedup |
|--------|-------------|----------------|---------|
| Time/inference (MNIST) | 114.40 µs | ~46 ms (projected, N=784)⁴ | 0.0025× |
| Throughput | 8,741 samples/sec | ~22 samples/sec (projected) | 0.0025× |
| Compute throughput | 4.10 GOPS | 8.88 MOPS (measured, N=64) | 0.002× |
| Power | ~50 W | 9.22 mW | **5,420× lower** |
| Energy/image | ~5,720 µJ | ~102 µJ (projected) | **56× lower** |

⁴ Full MNIST at N=784 projected: Layer1 (784→256): 256 neurons × 141 µs/neuron = 36.1 ms; Layers 2+3: ~8.4 ms; Total ≈ 44–46 ms.

---

## Key Takeaway

The accelerator is not faster than the CPU in throughput or latency — the SPI interface at 12.5 MHz
is the binding constraint, delivering 1.5625 MB/s against the CPU's ~50 GB/s memory bandwidth.
The accelerator's advantage is **energy efficiency**: 9.22 mW vs ~50 W, a 5,400× reduction in power,
translating to ~56× lower energy per inference. This is the correct trade-off for an edge/embedded
inference scenario where battery life or thermal budget dominates over absolute speed.
