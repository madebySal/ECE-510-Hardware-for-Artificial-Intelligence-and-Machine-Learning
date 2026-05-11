# Milestone 2 — BNN Inference Accelerator

## How to reproduce the M2 simulations

### Requirements

| Tool | Version tested |
|------|---------------|
| Icarus Verilog (`iverilog` / `vvp`) | 12.0 |
| Python | 3.12 |
| matplotlib | any recent |

On Windows the Icarus installer places `iverilog.exe` and `vvp.exe` in `C:\iverilog\bin\`.
Add that directory to your `PATH` before running the commands below.

### Compute core testbench

```bash
# from project/m2/
iverilog -g2012 -o sim_cc tb/tb_compute_core.sv rtl/compute_core.sv
vvp sim_cc
```

Expected output ends with `PASS`.

### Interface testbench

```bash
# from project/m2/
iverilog -g2012 -o sim_iface tb/tb_interface.sv rtl/interface.sv
vvp sim_iface
```

Expected output ends with `PASS`.

### Top-level integration testbench (bnn_top)

```bash
# from project/m2/
iverilog -g2012 -o sim_top tb/tb_bnn_top.sv rtl/bnn_top.sv rtl/interface.sv rtl/compute_core.sv
vvp sim_top
```

Runs 4 end-to-end SPI tests (all_agree, all_disagree, mixed, half_agree).
Expected output ends with `ALL PASS`.

### Waveform

```bash
# from project/m2/
python gen_waveform.py
```

Produces `sim/waveform.png`.

### Quantization error analysis

```bash
# from project/m2/
python sim/quantization_analysis.py
```

Runs 100 random test vectors through the reference BNN model and the hardware model.
Produces `sim/quantization_results.csv`. Expected: MAE=0, max error=0, 100% match.

---

## Deviations from M1 plan

**Interface:** no change. SPI Mode 0 as selected in `project/m1/interface_selection.md`.

**Module naming note:** SystemVerilog reserves `interface` as a keyword.
`project/m2/rtl/interface.sv` uses the escaped identifier `\interface` (valid
per IEEE 1800-2017 §5.6), so the top module name matches the filename exactly.
The SPI protocol and register map are identical to `project/hdl/spi_slave.sv`.

**Testbench format:** M1 development used Python/cocotb testbenches
(`project/hdl/test_bnn_layer.py`, `project/hdl/test_spi_slave.py`).
M2 delivers standalone SystemVerilog testbenches (`tb/tb_compute_core.sv`,
`tb/tb_interface.sv`) that run under plain `iverilog`/`vvp` without any
Python dependency. The cocotb versions remain in `project/hdl/` for reference.

**Compute core:** `project/m2/rtl/compute_core.sv` is the same XNOR-popcount
design as `project/hdl/bnn_layer.sv`; module renamed from `bnn_layer` to
`compute_core` to match the required filename.
