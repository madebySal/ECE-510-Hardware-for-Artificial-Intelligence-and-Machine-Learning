# 2×2 Weight-Stationary Systolic Array Trace
`codefest/cf05/cman_systolic_trace`

**Problem:** Compute C = A × B  
**A** = [[1, 2], [3, 4]] &nbsp; **B** = [[5, 6], [7, 8]] &nbsp; **Expected C** = [[19, 22], [43, 50]]

---

## Task 1 — PE Diagram (Preloaded Weights)

```
              col 0              col 1
            ┌──────────────┐  ┌──────────────┐
row 0 ──→   │  PE[0][0]    │  │  PE[0][1]    │
            │  w = B[0][0] │  │  w = B[0][1] │
            │    = 5       │  │    = 6       │
            │ MAC: a·5+ps↑ │  │ MAC: a·6+ps↑ │
            └──────┬───────┘  └──────┬───────┘
                   ↓ psum             ↓ psum
            ┌──────────────┐  ┌──────────────┐
row 1 ──→   │  PE[1][0]    │  │  PE[1][1]    │
            │  w = B[1][0] │  │  w = B[1][1] │
            │    = 7       │  │    = 8       │
            │ MAC: a·7+ps↑ │  │ MAC: a·8+ps↑ │
            └──────────────┘  └──────────────┘
```

**Dataflow rules (weight-stationary):**
- Weights (B values) are **pre-loaded and fixed** in each PE
- Inputs (A rows) **stream left → right** across each row
- Partial sums **accumulate downward** (↓) from row 0 → row 1
- Each PE computes: `psum_out = (input × weight) + psum_in`
- Row 1 inputs are **skewed by 1 cycle** so data aligns correctly at each PE

---

## Task 2 — Cycle-by-Cycle Table (4 Cycles)

> **Input skewing:** A row 0 = [a00=1, a01=2] enters at cycles 1 & 2.  
> A row 1 = [a10=3, a11=4] enters at cycles 2 & 3 (delayed by 1 cycle).

| Cycle | In→row0 (a0x) | In→row1 (a1x, skewed) | PE[0][0] psum | PE[0][1] psum | PE[1][0] psum         | PE[1][1] psum         | Output C            |
|-------|---------------|----------------------|---------------|---------------|-----------------------|-----------------------|---------------------|
| 1     | a00 = 1       | — (0)                | 1×5 = **5**   | 1×6 = **6**   | 0 + 0×7 = 0           | 0 + 0×8 = 0           | —                   |
| 2     | a01 = 2       | a10 = 3              | 2×5 = **10**  | 2×6 = **12**  | 5 + 3×7 = **26**      | 6 + 3×8 = **30**      | —                   |
| 3     | — (0)         | a11 = 4              | 0×5 = 0       | 0×6 = 0       | 10 + 4×7 = **38**     | 12 + 4×8 = **44**     | —                   |
| 4     | —             | —                    | 0             | 0             | 26+0 → **C[0][0]=19** | 30+0 → **C[0][1]=22** | **C=[[19,22],[43,50]]** |

**Accumulation detail:**

Each output value C[i][j] accumulates across two cycles at PE[1][j]:

| Output  | Computation                    | Result |
|---------|-------------------------------|--------|
| C[0][0] | 1×5 + 2×7 = 5 + 14           | **19** ✓ |
| C[0][1] | 1×6 + 2×8 = 6 + 16           | **22** ✓ |
| C[1][0] | 3×5 + 4×7 = 15 + 28          | **43** ✓ |
| C[1][1] | 3×6 + 4×8 = 18 + 32          | **50** ✓ |

---

## Task 3 — MAC & Memory Access Counts

### (a) Total MAC Operations: **8**

| PE        | Weight | Inputs processed | MACs |
|-----------|--------|-----------------|------|
| PE[0][0]  | 5      | a00=1, a01=2    | 2    |
| PE[0][1]  | 6      | a00=1, a01=2    | 2    |
| PE[1][0]  | 7      | a10=3, a11=4    | 2    |
| PE[1][1]  | 8      | a10=3, a11=4    | 2    |
| **Total** |        |                 | **8**|

Formula: 4 output values × 2 multiply-accumulates each = **8 MACs**

---

### (b) Input Value Reuse: **2× reuse**

- Each **A input value** is loaded once off-chip but passes through **2 PEs** as it streams across a row → **2× reuse**
- Each **B weight** stays fixed in its PE and is applied to **2 different input values** (one per input) → **2× reuse**

Weight-stationary dataflow is specifically designed to maximize weight reuse — weights never move after the initial load.

---

### (c) Off-Chip Memory Accesses

| Matrix | Values | Accesses | Detail                                      |
|--------|--------|----------|---------------------------------------------|
| A      | 4      | **4**    | a00, a01, a10, a11 — each loaded once       |
| B      | 4      | **4**    | Loaded once at startup, then stay on-chip   |
| C      | 4      | **4**    | Written once when all MACs complete         |

> **Key advantage:** B weights are loaded **once** and never re-fetched from off-chip memory. This is the primary benefit of weight-stationary dataflow — it eliminates repeated B memory reads that would otherwise dominate energy consumption.

---

## Task 4 — Output-Stationary Dataflow (One-Sentence Answer)

In an **output-stationary** dataflow, the **partial sums (accumulator values)** for each output element C[i][j] remain fixed inside their assigned PE throughout the entire computation, while inputs and weights stream through the array.

---

*2×2 Weight-Stationary Systolic Array · A×B=C · codefest/cf05*
