# Synthesis Interpretation — compute_core (BNN XNOR-Popcount)

**Tool:** Yosys 0.36+42 + ABC, sky130_fd_sc_hd tt/25°C/1.8V | **Clock target:** 10 ns (100 MHz)

## Clock Period and Worst-Case Slack

ABC reported a combinational critical path of **4498.91 ps (4.499 ns)**, giving a setup
slack of **+5.501 ns**. Timing is comfortably met; the design could sustain ~222 MHz
before violation. No hold violations exist because all outputs are registered with no
internal pipeline stages to create hold-sensitive paths.

## Critical Path

- **Source:** Primary input `weight_row[1]` (no source register — inputs arrive combinationally)
- **Sink:** `out` flip-flop (`sky130_fd_sc_hd__dfxtp_1`) D-pin
- **Dominant stages:** First XNOR (`xnor2_4`, 143 ps) → five `xnor3_1` reduction stages
  (~400 ps each, cumulative 2567 ps) → two `xor3_1` accumulation stages (2975–3394 ps) →
  comparator chain (`nor2_2`, `a21oi_2`, `o21ai_2`, `or3_1`, `isobufsrc_1`, `a211oi_1`,
  total 1105 ps) → DFF-D.
- The **XOR/XNOR reduction tree** for the 64-bit popcount accounts for the first 3.4 ns,
  making it the bottleneck.

## Total Cell Area and Top Three Contributors

**Total area: 2874.01 µm²**, **259 cells** (257 combinational + 2 DFFs).

| Rank | Cell | Count | Why dominant |
|------|------|-------|-------------|
| 1 | `xor2_1` | 59 | Low-level XOR pairs throughout popcount tree |
| 2 | `maj3_1` + `maj3_2` | 45 | Carry generation in adder tree |
| 3 | `xnor3_1` + `xor3_1` | 48 | Mid-level reduction; larger cells, high area per instance |

## Warnings and Constraints

234 warnings (26 unique), all from `dfflibmap` reporting unsupported pin expressions on
scan-chain cells (`sdfxtp`, `sdfrbp`, etc.) in the sky130 liberty file. These cells were
never instantiated; warnings are safe to ignore. The SDC `create_clock` token was not
recognized by ABC, so timing is an ABC-internal estimate rather than a full OpenSTA result.
No constraint failures or hold violations.
