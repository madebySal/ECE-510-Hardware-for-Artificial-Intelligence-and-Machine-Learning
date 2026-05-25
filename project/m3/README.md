# Milestone 3 — BNN Inference Accelerator

## File Catalog

| File | Description |
|------|-------------|
| `README.md` | This file — catalogs all M3 deliverables |
| `rtl/top.sv` | Integrated top module; instantiates M2 `\interface` + `compute_core`, connects all inter-module signals |
| `tb/tb_top.sv` | End-to-end co-simulation testbench; drives SPI protocol only (no direct core access); 4 test vectors with independent reference model |
| `sim/cosim_run.log` | Co-simulation transcript from `vvp sim_top` — shows ALL PASS (4/4 tests) |
| `sim/cosim_waveform.png` | End-to-end waveform annotated with ① host SPI write, ② compute activity, ③ host SPI read |
| `sim/sim_top` | Compiled simulation binary (iverilog output) |
| `sim/cosim.vcd` | VCD waveform dump from simulation |
| `sim/gen_cosim_waveform.py` | Python script that generates `cosim_waveform.png` |
| `synth/config.json` | OpenLane 2 configuration: design name, source files, clock period 10 ns, PDK sky130A |
| `synth/constraints.sdc` | SDC timing constraints for OpenLane 2 |
| `synth/openlane_run.log` | Full OpenLane 2 run log — includes actual ImportError and diagnosis |
| `synth/timing_report.txt` | STA report (analytical estimate; synthesis failed on Windows — see notes) |
| `synth/area_report.txt` | Area / cell count (analytical estimate; synthesis failed on Windows) |
| `synth/power_report.txt` | Power estimate (analytical estimate; power flow attempted, failed same cause) |
| `synth/critical_path.md` | Critical path identification: start/end points, 6 logic stages, explanation, mitigation |
| `synthesis_notes.md` | Narrative (≥500 words): what synthesised (RTL), what failed (OpenLane/Windows SIGKILL), scope status, M4 path forward |

---

## How to Reproduce the Co-Simulation

### Requirements

| Tool | Version |
|------|---------|
| Icarus Verilog (`iverilog` / `vvp`) | 12.0 |
| Python | 3.12 |
| matplotlib | any recent |

On Windows: Icarus installer places binaries in `C:\iverilog\bin\` — add to `PATH`.

### Compile and run

```bash
# from project/m3/
iverilog -g2012 -o sim/sim_top \
  tb/tb_top.sv \
  rtl/top.sv \
  ../m2/rtl/interface.sv \
  ../m2/rtl/compute_core.sv

vvp sim/sim_top
```

Expected output:
```
PASS  all_agree:    out=1 expected=1
PASS  all_disagree: out=0 expected=0
PASS  mixed:        out=1 expected=1
PASS  half_agree:   out=0 expected=0
----------------------------------------
ALL PASS  (4/4 tests passed)
----------------------------------------
```

### Regenerate waveform

```bash
# from project/m3/sim/
python gen_cosim_waveform.py
```

Produces `sim/cosim_waveform.png`.

---

## How to Reproduce the OpenLane 2 Synthesis Run

### OpenLane 2 version and configuration

| Item | Value |
|------|-------|
| OpenLane version | 2.3.10 (pip install openlane) |
| PDK | sky130A / sky130_fd_sc_hd |
| Clock period | 10.0 ns (100 MHz) |
| Config file | `project/m3/synth/config.json` |
| Constraints | `project/m3/synth/constraints.sdc` |

### Status on this platform

OpenLane 2 cannot run natively on Windows 11 (see `synth/openlane_run.log`).
The tool fails with `ImportError: cannot import name 'SIGKILL'` because it uses
Linux-specific POSIX signals internally.

### Reproduction command (requires Docker Desktop running)

```bash
# Start Docker Desktop first, then from project/m3/:
docker run --rm \
  -v "$(pwd):/work" \
  ghcr.io/efabless/openlane2:latest \
  openlane /work/synth/config.json
```

Reports will be written to `runs/RUN_<timestamp>/reports/`.

### Environment variables required

```bash
# None required beyond Docker Desktop being active.
# If PDK_ROOT is not set, OpenLane 2 will download sky130A automatically.
# To use a pre-installed PDK:
export PDK_ROOT=/path/to/pdks   # optional
```

---

## Deviations from M2

- **`top.sv`** is identical in function to `project/m2/rtl/bnn_top.sv` but renamed
  to match the M3 required filename (`top.sv`) and includes an expanded header
  comment documenting all ports, data flow, and glue logic rationale.
- M2 RTL files (`interface.sv`, `compute_core.sv`) are referenced in-place from
  `project/m2/rtl/` — no duplication.
- M3 testbench uses a module-level shared buffer (`tx_buf[0:15]`) to work around
  an Icarus Verilog limitation with open-array parameters in automatic tasks.
