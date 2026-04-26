# 4×4 FP32 Weight Matrix Quantization

## Given Matrix W

```
W = [[ 0.85, -1.20,  0.34,  2.10],
     [-0.07,  0.91, -1.88,  0.12],
     [ 1.55,  0.03, -0.44, -2.31],
     [-0.18,  1.03,  0.77,  0.55]]
```

---

## Task 1: Scale Factor

**Formula:** S = max(|W|) / 127

Finding |W| for all elements:

```
|W| = [[0.85, 1.20, 0.34, 2.10],
       [0.07, 0.91, 1.88, 0.12],
       [1.55, 0.03, 0.44, 2.31],
       [0.18, 1.03, 0.77, 0.55]]
```

**max(|W|) = 2.31** (element at row 3, col 4)

$$S = \frac{2.31}{127} \approx \boxed{0.018189}$$

---

## Task 2: Quantize → W\_q = round(W / S), clamped to [−128, 127]

W / S (dividing each element by 0.018189):

```
W/S ≈ [[ 46.73, -65.97,  18.69, 115.45],
        [ -3.85,  50.03, -103.36,  6.60],
        [ 85.22,   1.65, -24.19, -127.00],
        [ -9.90,  56.63,  42.33,  30.24]]
```

After `round()` and clamp to [−128, 127]:

```
W_q = [[ 47, -66,  19, 115],
       [ -4,  50, -103,   7],
       [ 85,   2, -24, -127],
       [-10,  57,  42,  30]]
```

> All values fall within [−128, 127] — no clamping needed here.

---

## Task 3: Dequantize → W\_deq = W\_q × S

Multiplying each INT8 value by S = 0.018189:

```
W_deq = [[ 0.8549, -1.2005,  0.3456,  2.0917],
          [-0.0728,  0.9095, -1.8735,  0.1273],
          [ 1.5461,  0.0364, -0.4365, -2.3100],
          [-0.1819,  1.0368,  0.7639,  0.5457]]
```

---

## Task 4: Error Analysis

**Per-element absolute error |W − W\_deq|:**

```
Error = [[0.0049, 0.0005, 0.0056, 0.0083],
          [0.0028, 0.0005, 0.0065, 0.0073],
          [0.0039, 0.0064, 0.0035, 0.0000],
          [0.0019, 0.0068, 0.0061, 0.0043]]
```

**Largest error:** 0.0083 → at position (row 1, col 4), element W = **2.10**

**MAE** (mean of all 16 errors):

$$MAE = \frac{\sum|W - W_{deq}|}{16} = \frac{0.0693}{16} \approx \boxed{0.00433}$$

---

## Task 5: Bad Scale Experiment (S\_bad = 0.01)

**W / S\_bad** (dividing by 0.01):

```
W/S_bad = [[ 85, -120,  34, 210],
            [ -7,  91, -188,  12],
            [155,   3,  -44, -231],
            [-18, 103,   77,  55]]
```

After `round()` and **clamp to [−128, 127]**:

```
W_q_bad = [[ 85, -120,  34, 127],   ← 210 clamped to 127
            [ -7,  91, -128,  12],   ← -188 clamped to -128
            [127,   3,  -44, -128],  ← 155→127, -231→-128
            [-18, 103,   77,  55]]
```

**W\_deq\_bad = W\_q\_bad × 0.01:**

```
W_deq_bad = [[ 0.85, -1.20,  0.34,  1.27],
              [-0.07,  0.91, -1.28,  0.12],
              [ 1.27,  0.03, -0.44, -1.28],
              [-0.18,  1.03,  0.77,  0.55]]
```

**Error |W − W\_deq\_bad|:**

```
Error = [[0.00, 0.00, 0.00, 0.83],
          [0.00, 0.00, 0.60, 0.00],
          [0.28, 0.00, 0.00, 1.03],
          [0.00, 0.00, 0.00, 0.00]]
```

$$MAE_{bad} = \frac{2.74}{16} \approx \boxed{0.1713}$$

> ⚠️ **Compare:** MAE went from **0.00433 → 0.1713** (≈40× worse!)

**Explanation:** When S is too small, large-magnitude weights exceed the INT8 representable range and get **clamped**, causing severe irreversible information loss for those elements.
