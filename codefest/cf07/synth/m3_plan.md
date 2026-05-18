# M3 Synthesis Plan — compute_core (Option A)

The N=64 core meets timing with +5.501 ns slack at 10 ns, leaving substantial headroom.
For M3, the critical concern is **scaling to N=784** (production width), which extends
the popcount tree from ~7 to ~10 levels and is estimated to add ~2 ns to the 4.499 ns
path — still within a 10 ns constraint.

**Actions for M3:**

1. **Re-synthesize at N=784.** Area is expected to scale ~12× (from 2874 µm²) to ~35 K µm².
   If path exceeds 10 ns, pipeline after 5 XOR-tree levels (2-cycle latency).

2. **Include the SPI interface module.** The interface was excluded from CF07. At N=784 it
   holds 1568 flip-flops for activation/weight registers — likely dominating area in the
   full-chip synthesis.

3. **Run full OpenLane 2 on the school server** for post-route timing; the Yosys/ABC
   estimate here omits P&R wire-load and routing congestion effects.
