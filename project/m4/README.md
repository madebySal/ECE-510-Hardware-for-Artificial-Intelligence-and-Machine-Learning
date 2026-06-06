# Milestone 4 — BNN Inference Accelerator
## ECE 510 · Hardware for AI/ML · Spring 2026

This folder contains all M4 deliverables for the **BNN XNOR-Popcount Inference Accelerator**
implemented on sky130A / sky130_fd_sc_hd at 100 MHz.

**Design justification report:** [`report/design_justification.pdf`](report/design_justification.pdf)

---

## File Catalog

| File | Description | Checklist |
|------|-------------|-----------|
| `README.md` (this file) | Catalogs all M4 files with descriptions and checklist references | §1 README |

### RTL — `rtl/`
| File | Description | Checklist |
|------|-------------|-----------|
| `rtl/top.sv` | Top-level integration: \interface + compute_core, single clock domain | §2 Source code |
| `rtl/compute_core.sv` | XNOR array (N=64) + $countones popcount + threshold DFF | §2 Source code |
| `rtl/interface.sv` | SPI Mode-0 slave, 128-byte DFF register file, auto-increment addressing | §2 Source code |

> RTL is identical to M3. No changes between M3 and M4 submission.

### Testbench — `tb/`
| File | Description | Checklist |
|------|-------------|-----------|
| `tb/tb_top.sv` | End-to-end SPI co-simulation: 4 test vectors, independent ref_bnn(), self-contained | §2 Testbench |

Run command (from `project/m4/`):
```bash
iverilog -g2012 -o sim/sim_m4 tb/tb_top.sv rtl/top.sv rtl/interface.sv rtl/compute_core.sv
vvp sim/sim_m4
```

### Simulation — `sim/`
| File | Description | Checklist |
|------|-------------|-----------|
| `sim/final_run.log` | Icarus Verilog co-simulation output — 4/4 PASS | §2 Sim log |
| `sim/final_waveform.png` | Annotated end-to-end waveform: SPI write → compute → readback | §2 Waveform |

### Synthesis — `synth/`
| File | Description | Checklist |
|------|-------------|-----------|
| `synth/config.json` | OpenLane 2.3.10 config: sky130A, 10 ns clock, AREA 0 strategy | §3 Config |
| `synth/openlane_run.log` | Full OpenLane stdout: 78/78 steps, exit 0, 21m31s | §3 Run log |
| `synth/timing_report.txt` | Post-PNR STA: WNS +4.165 ns (nom), −1.915 ns (ss worst) | §3 Timing |
| `synth/area_report.txt` | 5,772 cells, 67,353 µm², 45.1% utilisation | §3 Area |
| `synth/power_report.txt` | 9.216 mW total (sequential 51.6%, clock 45.5%, comb 2.9%) | §3 Power |

### Benchmark — `bench/`
| File | Description | Checklist |
|------|-------------|-----------|
| `bench/benchmark.md` | HW vs SW throughput, speedup, energy comparison | §4 Benchmark |
| `bench/benchmark_data.csv` | Raw numbers behind all reported metrics | §4 Raw data |
| `bench/roofline_final.png` | Final roofline: sky130A + CPU baselines + measured M4 point | §4 Roofline |

### Report — `report/`
| File | Description | Checklist |
|------|-------------|-----------|
| `report/design_justification.pdf` | 9-section design justification, ~2,800 words | §5 Report |
| `report/figures/fig1_block_diagram.png` | System block diagram (§4 Dataflow) | §5 Figures |
| `report/figures/fig2_dataflow.png` | Activation-stationary dataflow (§4 Dataflow) | §5 Figures |
| `report/figures/fig3_roofline.png` | Final roofline plot (§2 Roofline, §8 Benchmark) | §5 Figures |
| `report/figures/fig4_waveform.png` | Co-simulation waveform (§6 Verification) | §5 Figures |

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Technology | sky130A / sky130_fd_sc_hd |
| Clock | 100 MHz |
| Setup slack (nominal tt 25°C 1.8V) | +4.165 ns ✓ |
| Total cells | 5,772 |
| Die area | 67,353 µm² (0.163 mm²) |
| Power (nominal) | 9.22 mW |
| Compute throughput | 8.88 MOPS (measured, co-sim) |
| Energy per inference | ~102 µJ (N=784 projected) |
| SW baseline energy | ~5,720 µJ |
| Energy improvement | **56× more efficient** |
