# Remaining Tasks Before M4
## ECE 510 Spring 2026 — BNN Accelerator Project

Listed in order of impact.

---

### Task 1 — Replace the SPI data interface with a 32-bit AXI4-Lite bus

**Specific change:** Rewrite `m2/rtl/interface.sv` to accept 32-bit word writes over AXI4-Lite
(`AWADDR`, `WDATA`, `WSTRB`, `BVALID`) instead of the current bit-serial SPI shift register.
Keep the same 128-byte logical register map; map each 4-byte-aligned word to four consecutive
register-file locations.

**Why:** The SPI interface at 12.5 MHz delivers 1.5625 MB/s. The AXI4-Lite at 100 MHz delivers
400 MB/s — a 256× increase. This moves the ridge point from 8,192 FLOPs/byte down to 32 FLOPs/byte,
placing the kernel's arithmetic intensity (8–16 FLOPs/byte) near the compute-bound region and
enabling the accelerator to approach its 12.8 GOPS peak instead of the current 8.88 MOPS.

---

### Task 2 — Add pipeline register to break the SPI regfile write-enable decode critical path

**Specific change:** In `m2/rtl/interface.sv`, insert one pipeline flip-flop between the 7-bit
address comparator output (`byte_sel[127:0]`) and the regfile write-mux input. The current
9-stage combinational cone (mosi_s1 → or4bb → nor2 → a22o → a221o → or4 × 3 → a221o → o221a)
takes 5.77 ns combinational delay, leaving +4.17 ns slack at nominal but failing at the slow corner
(WNS = −1.915 ns at max_ss_100C_1v60). Pipelining splits this into two ≤3 ns stages, meeting
all nine timing corners including the worst-case slow corner.

**Why:** The design currently has 9 setup violations at the slow corner. M4 synthesis must be
clean across all corners per the rubric. The pipeline add costs one extra SPI write latency cycle
(transparent to the host since SPI bursts are atomic at the byte granularity the host already uses).

---

### Task 3 — Extend the testbench to cover N = 784 inputs via parameter override

**Specific change:** In `m3/tb/tb_top.sv`, parameterize `N` and add a second test suite
compiled with `N = 784` using `iverilog -pN=784`. Add 3 test vectors at N = 784: all-agree,
all-disagree, and a checkerboard pattern (alternating 1/0 activations and weights, expected
pop = N/2, output = 0). Document the expected vs. actual output for each.

**Why:** All co-simulation and synthesis results were produced at N = 64. The production MNIST
target is N = 784. The M4 rubric requires demonstrating that the design scales to the production
point; without an N = 784 testbench, the claim rests on analysis alone. The RTL is parameterized
(`#(parameter int N = 64)`) so no RTL changes are needed — only testbench additions and a second
OpenLane synthesis run with `SYNTH_PARAMETERS: "N=784"` in config.json.
