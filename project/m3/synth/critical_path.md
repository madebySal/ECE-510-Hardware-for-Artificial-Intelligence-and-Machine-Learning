# Critical Path Analysis — BNN top (N=64)
## Source: OpenLane 2.3.10 / OpenROAD post-PNR STA, nom_tt_025C_1v80

## Summary

**Start point:** `_09152_` — `u_interface.mosi_s1`
(the second flip-flop of the 2-FF MOSI synchroniser chain in `\interface`)

**End point:** `_10196_` — one bit of the SPI register file (`regfile[*][*]`)
(a `dfxtp_1` flip-flop storing one bit of an incoming SPI data byte)

**Slack:** +4.165 ns (TIMING MET at nom_tt_025C_1v80)

---

## Logic Stages on the Critical Path

```
[CLK→Q]  _09152_  sky130_fd_sc_hd__dfxtp_4         0.000 ns → 1.553 ns
          Signal: u_interface.mosi_s1
             The synchronised MOSI bit, fully valid and stable.
             High fan-out (39 loads) requires a dfxtp_4 (drive-4) cell.

[BUF×2]  max_cap27, max_cap26  sky130_fd_sc_hd__buf_12    1.553 ns → 2.390 ns
          Two cascaded buf_12 cells rebuffer mosi_s1 to 64 downstream loads.
          Net cap on mosi_s1 after the first buf_12 is still 0.50 pF;
          a second buf_12 is needed before the decode tree.

[LOGIC1] _07157_  sky130_fd_sc_hd__or4bb_4           2.390 ns → 2.968 ns
          or4bb = OR of 4 inputs with two inverting inputs.
          Part of the 7-bit current address (cur_addr[6:0]) match tree:
          checks which register address is currently being written.

[LOGIC2] _07225_  sky130_fd_sc_hd__nor2_8             2.968 ns → 3.212 ns
          High-drive NOR2 (drive-8) combining two address decode signals.

[LOGIC3] _07489_  sky130_fd_sc_hd__a22o_1             3.212 ns → 3.905 ns
          AND-AND-OR (a22o): combines mosi data bit with write-enable.
          This gate is the join point where incoming data meets the
          decoded address; it is the write-data multiplexer for the regfile.

[LOGIC4] _07490_  sky130_fd_sc_hd__a221o_1            3.905 ns → 4.352 ns
[LOGIC5] _07493_  sky130_fd_sc_hd__or4_1              4.352 ns → 4.842 ns
[LOGIC6] _07498_  sky130_fd_sc_hd__or4_1              4.842 ns → 5.386 ns
          Three more levels of OR4 / compound gates propagating the
          byte-granular write-enable through the 128-entry regfile column
          select logic. Each or4_1 adds ~0.54 ns.

[LOGIC7] _07531_  sky130_fd_sc_hd__or4_4              5.386 ns → 5.999 ns
          or4_4 (drive-4) for the final register bank select.

[LOGIC8] _07533_  sky130_fd_sc_hd__a221o_1            5.999 ns → 6.328 ns
[LOGIC9] _07534_  sky130_fd_sc_hd__o221a_1            6.328 ns → 6.530 ns
          Final two compound gates before the register D-input.

[Setup]  _10196_  sky130_fd_sc_hd__dfxtp_1            6.530 ns → 6.628 ns
          Library setup time = 0.098 ns.

  Required time (10 ns period + clock skew adjustments): 10.696 ns
  Arrival time:                                            6.530 ns
  SLACK:                                                  +4.165 ns ✓
```

---

## Why This Is the Critical Path

The critical path is **not** through the XNOR+popcount compute core (as predicted in the M3 analytical estimate). The compute core's `$countones` popcount synthesised to a clean adder tree (~3.5 ns) which has ample slack. The actual bottleneck is the **SPI register-file write-path decode logic** in `\interface`.

The 128-byte register file requires:
1. A 7-bit address comparator (which 1-of-128 bytes is being written)
2. A byte-granular write-enable signal propagated to 8 flip-flops
3. A data-mux that routes the shifted-in MOSI bit to the selected flip-flop

Steps 1–3 create a 9-stage logic cone (BUF→OR4bb→NOR2→A22O→A221O→OR4→OR4→OR4→A221O→O221A) with a total combinational delay of ~5.0 ns. With CLK→Q (1.55 ns) and setup (0.098 ns), the path total is 6.63 ns, leaving +4.17 ns of slack against a 10 ns clock.

The MOSI synchroniser flip-flop (`mosi_s1`) is the start point because it directly drives the write-data path. Every incoming SPI bit must traverse this entire decode tree to reach the targeted regfile DFF in a single system clock cycle.

---

## Why It Fails at the Slow Corner (max_ss_100C_1v60)

At 100°C and 1.60 V (versus 25°C and 1.80 V at nominal):
- Cell delays increase ~1.83× across the chain
- The same 9-stage decode path takes ~11.9 ns (arrival), exceeding the 10 ns clock period
- WNS = −1.915 ns (27 paths violating)

The slow corner is a stress test, not an operating condition for this design; the target MNIST inference accelerator is expected to operate near nominal conditions.

---

## What Would Shorten the Critical Path

1. **Register the write-enable decode.** Insert a pipeline register between the address decode and the regfile write mux. This splits the 9-stage cone into two 4–5 stage cones at the cost of one additional latency cycle for SPI writes (transparent to the host since the burst is still atomic).

2. **Reduce the register file.** The 128-byte regfile is larger than necessary: only 8+8 = 16 bytes of activation + weight are needed, plus 2 control bytes. Reducing to 32 entries (5-bit address) eliminates two levels of OR4 decode, saving ~1.1 ns.

3. **Reduce the clock to 50 MHz.** The 10 ns constraint is tighter than the SPI bus bottleneck. At 12.5 MHz SPI the system clock only needs to be fast enough to complete SPI sampling (8× oversampling → 12.5 MHz × 8 = 100 MHz minimum). At 50 MHz the design would comfortably meet all corners.

4. **Use a smaller PDK.** Migrating to a 22 nm or 28 nm process would reduce cell delays by 3–4× and push the critical path well below 3 ns.
