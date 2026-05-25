# Critical Path Analysis — BNN top (N=64)

## Summary

**Start point:** `u_compute_core / activation[*]` (the activation register
output in `compute_core`, clocked by `clk`)

**End point:** `u_compute_core / out` register D-input
(the neuron output flip-flop in `compute_core`)

**Logic stages on the critical path (6 stages):**

```
[CLK→Q]  activation[] DFF output
   ↓
[Stage 1]  64× XNOR2: xnor_vec[i] = ~(activation[i] ^ weight_row[i])
   ↓
[Stage 2]  Adder tree level 1: 32 full-adders, 64 single bits → 32 × 2-bit partial sums
   ↓
[Stage 3]  Adder tree level 2: 16 full-adders, 32 partial sums → 16 × 3-bit sums
   ↓
[Stage 4]  Adder tree levels 3–5: reduce to a single 6-bit popcount value
   ↓
[Stage 5]  Comparator: (pop[6:0] > 7'd32) — 7-bit unsigned magnitude compare
   ↓
[Stage 6]  MUX (ternary select): 1'b1 if compare true, else 1'b0
   ↓
[Setup]   out DFF setup time
```

---

## Why This Is the Critical Path

The BNN neuron has two parallel computation branches:
1. The XNOR array (64 independent gate pairs — no carry chain, fast)
2. The **popcount adder tree** (reduces 64 bits to a 7-bit sum — deep carry structure)

Branch 2 is dominant. A 64-input popcount requires 6 levels of binary addition
(log₂64 = 6). Each level adds one full-adder delay. The adder tree is the
longest logic cone in the entire design; the SPI register file and FSM in
`\interface` are controlled by the slow SPI clock and do not appear on the
system-clock critical path.

---

## Gate-Level Timing Breakdown (sky130_fd_sc_hd, typical 1.8 V 25 °C)

| Stage | Logic | # Cells | Est. delay |
|-------|-------|---------|-----------|
| CLK→Q | DFF (sky130_fd_sc_hd__dfxtp_1) | 1 | 0.20 ns |
| XNOR | sky130_fd_sc_hd__xnor2_1 | 64 | 0.15 ns |
| Tree L1 | sky130_fd_sc_hd__fa_1 (×32) | 32 | 0.50 ns |
| Tree L2 | sky130_fd_sc_hd__fa_1 (×16) | 16 | 0.50 ns |
| Tree L3–L5 | sky130_fd_sc_hd__fa_1 (×15) | 15 | 1.50 ns |
| Tree L6 | sky130_fd_sc_hd__fa_1 (×1, final) | 1 | 0.50 ns |
| Comparator | sky130_fd_sc_hd__a21oi / o21ai | ~10 | 0.40 ns |
| FF setup | sky130_fd_sc_hd__dfxtp_1 | 1 | 0.10 ns |
| **Total** | | | **~3.85 ns** |

At a 10 ns clock period, the estimated worst negative slack is **+6.15 ns**,
meaning the design closes timing comfortably at 100 MHz and could be pushed
to approximately **260 MHz** before the adder tree becomes the timing bottleneck.

---

## What Would Shorten the Critical Path

1. **Replace the ripple adder tree with a carry-save adder (CSA) tree.**
   CSA reduces carry propagation from O(log N) stages to a constant 3 stages
   (partial products + final carry-propagate adder), cutting the tree delay
   from ~3.5 ns to ~1.0 ns.

2. **Pipeline the popcount.** Insert a register after the adder tree, splitting
   the combinational cone across two cycles. The throughput would remain one
   result per clock because the pipeline fills in the second cycle.

3. **Use a dedicated popcount LUT.** For N=64 the XNOR result is 64 bits; split
   into four 16-bit words, look up each in a pre-computed 5-bit ROM, then add
   the four 5-bit values. Only one level of 5-bit addition remains.

4. **Change the technology target.** Migrating from sky130A (130 nm, ~0.5 ns/FA)
   to a 22 nm PDK would reduce the adder tree delay by ~3×, yielding a
   sub-1 ns critical path and 1+ GHz operation from the same RTL.

For the M1 target of 10,000 inferences per second (100 µs/inference budget),
the current design has 6× timing headroom even at the 100 MHz constraint.
The critical path is not a bottleneck at this throughput target.
