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

### Waveform

```bash
# from project/m2/
python gen_waveform.py
```

Produces `sim/waveform.png`.

---

## Deviations from M1 plan

**Interface:** no change. SPI Mode 0 as selected in `project/m1/interface_selection.md`.

**Module naming note:** SystemVerilog reserves the keyword `interface`, so
`project/m2/rtl/interface.sv` declares its top module as `interface_module`.
The SPI protocol and register map are identical to `project/hdl/spi_slave.sv`
from M1 development; only the module name changed to satisfy the synthesizable
naming requirement.

**Testbench format:** M1 development used Python/cocotb testbenches
(`project/hdl/test_bnn_layer.py`, `project/hdl/test_spi_slave.py`).
M2 delivers standalone SystemVerilog testbenches (`tb/tb_compute_core.sv`,
`tb/tb_interface.sv`) that run under plain `iverilog`/`vvp` without any
Python dependency. The cocotb versions remain in `project/hdl/` for reference.

**Compute core:** `project/m2/rtl/compute_core.sv` is the same XNOR-popcount
design as `project/hdl/bnn_layer.sv`; module renamed from `bnn_layer` to
`compute_core` to match the required filename.
