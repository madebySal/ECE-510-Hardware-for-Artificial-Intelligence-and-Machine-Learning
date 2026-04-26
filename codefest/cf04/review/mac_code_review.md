# MAC Code Review — Codefest 4

## LLM Versions
| File | Model |
|------|-------|
| mac_llm_A.v | Claude Sonnet 4.6 |
| mac_llm_B.v | GPT-4o (gpt-4o-2024-11-20) |

---

## Compilation Results

```
iverilog -g2012 -o mac_a mac_llm_A.v mac_tb.v   → exit 0 (no errors)
iverilog -g2012 -o mac_b mac_llm_B.v mac_tb.v   → exit 0 (no errors)
iverilog -g2012 -o mac_c mac_correct.v mac_tb.v  → exit 0 (no errors)
```

---

## Simulation Results

### mac_llm_A (Claude Sonnet 4.6)
```
PASS cyc1: out=12
PASS cyc2: out=24
PASS cyc3: out=36
PASS reset: out=0
FAIL neg_cyc1: got 502 expected -10
FAIL neg_cyc2: got 1004 expected -20
```

### mac_llm_B (GPT-4o)
```
MAC reset          ← $display fires (non-synthesizable)
PASS cyc1: out=12
...
PASS neg_cyc1: out=-10
PASS neg_cyc2: out=-20
```

### mac_correct
```
PASS cyc1: out=12
PASS cyc2: out=24
PASS cyc3: out=36
PASS reset: out=0
PASS neg_cyc1: out=-10
PASS neg_cyc2: out=-20
```

---

## Issue 1 — Missing `signed` on port declarations (mac_llm_A.v)

**Offending lines:**
```verilog
input  logic [7:0]  a,
input  logic [7:0]  b,
output logic [31:0] out
```

**Why it's wrong:** Without `signed`, `a` and `b` are treated as unsigned logic vectors.
The expression `a * b` is therefore an unsigned 16-bit multiply. When `a = -5` (8'hFB = 251)
and `b = 2`, the result is 502 instead of -10. The bug only surfaces on negative inputs —
positive tests pass, giving false confidence.

**Corrected version:**
```systemverilog
input  logic signed [7:0]  a,
input  logic signed [7:0]  b,
output logic signed [31:0] out
```

---

## Issue 2 — No explicit sign extension on the product (mac_llm_A.v, mac_llm_B.v)

**Offending lines:**
```verilog
out <= out + (a * b);   // mac_llm_A
out <= out + a * b;     // mac_llm_B
```

**Why it's wrong:** `a * b` produces a 16-bit result. Adding it to a 32-bit accumulator
without an explicit cast relies on implicit sign extension rules that vary by tool and
synthesis target. In mac_llm_A this is moot (ports are unsigned anyway), but in mac_llm_B
it is an unguarded assumption. The safe, portable form casts the product explicitly.

**Corrected version:**
```systemverilog
out <= out + (32'(signed'(a)) * 32'(signed'(b)));
```
Widen both operands to 32 bits *before* multiplying. In SystemVerilog, `a * b` where both
are `[7:0]` produces only an 8-bit result (max of operand widths). The cast-after pattern
`32'(signed'(a * b))` sign-extends the already-truncated 8-bit product — which is wrong
for operands like 127×127 = 16129 that do not fit in 8 bits.

**Simulation evidence:** With the cast-before fix, `a=127, b=127` accumulates 16129 per cycle
and wraps at cycle 133145. With cast-after, the 8-bit product truncates to 1 and the
accumulator simply counts cycles.

---

## Issue 3 — Non-synthesizable constructs (mac_llm_B.v)

**Offending lines:**
```verilog
always @(posedge clk) begin   // should be always_ff
    $display("MAC reset");    // non-synthesizable
end

initial begin                 // non-synthesizable
    out = 32'sd0;
end
```

**Why it's wrong:**
- `always @(posedge clk)` is a behavioral construct; synthesis tools prefer (and lint tools
  require) `always_ff` for clocked registers so they can flag sensitivity-list errors.
- `$display` is a simulation-only system task; it will either be ignored or cause a synthesis
  error depending on the tool.
- `initial` blocks are not supported in most synthesis flows (exception: FPGA BRAM init).
  Using `initial` for reset is incorrect — reset must be done via the synchronous `if (rst)` path.

**Corrected version:**
```systemverilog
always_ff @(posedge clk) begin
    if (rst)
        out <= 32'sd0;
    else
        out <= out + 32'(signed'(a * b));
end
// remove initial block entirely
```
