# MAC Code Review — Codefest 4

## LLM Versions
| File | Model |
|------|-------|
| mac_llm_A.v | Claude Sonnet 4.6 |
| mac_llm_B.v | ChatGPT 5.3 |

---

## Compilation Results

```
iverilog -g2012 mac_llm_A.v mac_tb.v   → exit 0 (no errors)
iverilog -g2012 mac_llm_B.v mac_tb.v   → exit 0 (no errors)
iverilog -g2012 mac_correct.v mac_tb.v → exit 0 (no errors)
```

---

## Simulation Results

### mac_llm_A (Claude Sonnet 4.6)
```
PASS cyc1: out=12
PASS cyc2: out=24
PASS cyc3: out=36
PASS reset: out=0
PASS neg_cyc1: out=-10
PASS neg_cyc2: out=-20
PASS large_cyc1: out=10000
PASS large_cyc2: out=20000
PASS large_cyc3: out=30000
```

### mac_llm_B (ChatGPT 5.3)
```
PASS cyc1: out=12
PASS cyc2: out=24
PASS cyc3: out=36
PASS reset: out=0
PASS neg_cyc1: out=-10
PASS neg_cyc2: out=-20
PASS large_cyc1: out=10000
PASS large_cyc2: out=20000
PASS large_cyc3: out=30000
```

### mac_correct
```
PASS cyc1: out=12  ...  PASS large_cyc3: out=30000  (all 9 PASS)
```

---

## Issue 1 — Tool-dependent expression width in `mult = a * b` (mac_llm_B.v)

**Offending lines:**
```systemverilog
logic signed [15:0] mult;

always_comb begin
    mult = a * b;
end
```

**Why it is ambiguous:** The expression `a * b` where both operands are `logic signed [7:0]`
has a *self-determined* width of 8 bits under strict IEEE 1800 rules. When assigned to
`logic signed [15:0] mult`, some tools (including Icarus with `-g2012`) extend the context
to 16 bits and evaluate the multiplication at full precision — so the code passes in
simulation. However, other synthesis tools (e.g., Synopsys DC, Quartus) may evaluate
`a * b` in 8-bit self-determined context first, truncate the product, and then sign-extend
the truncated 8-bit result into `mult`. For `a = 100, b = 100`, the 8-bit truncated product
is 16 (100×100 = 10000, 10000 mod 256 = 16), not 10000 — a silent correctness failure
that only surfaces on large operands. The code passes simulation on Icarus but is not
portably correct.

**Corrected version:**
```systemverilog
always_comb begin
    mult = 16'(signed'(a)) * 16'(signed'(b));
end
```
Explicit casts widen `a` and `b` to 16 bits *before* the multiply, making the operation
unambiguous across all tools.

---

## Issue 2 — Unnecessary `always_comb` block (mac_llm_B.v)

**Offending lines:**
```systemverilog
logic signed [15:0] mult;

always_comb begin
    mult = a * b;
end

always_ff @(posedge clk) begin
    ...
    out <= out + mult;
end
```

**Why it is problematic:** The spec requires `always_ff` for the sequential logic and says
nothing about a combinational intermediate. Splitting the multiply into a separate
`always_comb` block adds an extra net (`mult`) and process, which:
- Creates an additional fanout path the synthesizer must analyze for timing closure.
- Obscures the intent: the multiply exists only to feed the accumulator, so it belongs
  inline in the `always_ff` block.
- Introduces a combinational glitch window on `mult` between clock edges that, while
  harmless here, is a design pattern to avoid in critical paths.

**Corrected version (inline multiply, no intermediate):**
```systemverilog
always_ff @(posedge clk) begin
    if (rst)
        out <= 32'sd0;
    else
        out <= out + 32'(signed'(a)) * 32'(signed'(b));
end
```

---

## Issue 3 — Implicit sign extension from 16-bit `mult` to 32-bit `out` (mac_llm_B.v)

**Offending line:**
```systemverilog
out <= out + mult;
```

**Why it is ambiguous:** `out` is `signed [31:0]` and `mult` is `signed [15:0]`. The addition
requires `mult` to be sign-extended to 32 bits. In Icarus this works correctly because both
signals are declared `signed`. However, mixing operand widths in accumulations without
explicit extension is a common source of synthesis mismatches: if `mult` were inadvertently
declared without `signed`, the zero-extension would silently produce wrong results for
negative products. Explicit casting makes intent clear and is safer across tools.

**Corrected version:**
```systemverilog
out <= out + 32'(signed'(mult));
```

---

## Summary

| File | Issues | Testbench result |
|------|--------|-----------------|
| mac_llm_A.v (Claude Sonnet 4.6) | None | All 9 PASS |
| mac_llm_B.v (ChatGPT 5.3) | 3 (portability, style, implicit extension) | All 9 PASS on Icarus; may fail on other tools |
| mac_correct.v | — | All 9 PASS |
