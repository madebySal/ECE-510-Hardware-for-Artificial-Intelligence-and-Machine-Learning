# CMAN — Workload Accounting
## ECE 510 Spring 2026 | Codefest 01
## Student: Saleh Esmaeil

**Network:** [784 → 256 → 128 → 10], batch size = 1, FP32 (4 bytes/value), no biases

---

## (a) MACs per Layer

**Formula:** MACs = input\_size × output\_size

| Layer | Calculation | MACs |
|-------|-------------|------|
| Layer 1 (784 → 256) | 784 × 256 | 200,704 |
| Layer 2 (256 → 128) | 256 × 128 | 32,768 |
| Layer 3 (128 → 10)  | 128 × 10  | 1,280  |

---

## (b) Total MACs

```
200,704 + 32,768 + 1,280 = 234,752 MACs
```

---

## (c) Total Parameters (weights only, no biases)

| Layer | Calculation | Parameters |
|-------|-------------|------------|
| Layer 1 | 784 × 256 | 200,704 |
| Layer 2 | 256 × 128 |  32,768 |
| Layer 3 | 128 × 10  |   1,280 |
| **Total** | | **234,752** |

---

## (d) Weight Memory (FP32)

```
234,752 parameters × 4 bytes = 939,008 bytes
```

---

## (e) Activation Memory (input + all layer outputs)

| Tensor | Calculation | Bytes |
|--------|-------------|-------|
| Input      | 784 × 4  | 3,136 |
| Layer 1 output | 256 × 4  | 1,024 |
| Layer 2 output | 128 × 4  |   512 |
| Layer 3 output |  10 × 4  |    40 |
| **Total** | | **4,712 bytes** |

---

## (f) Arithmetic Intensity

**Formula:** AI = (2 × total MACs) / (weight bytes + activation bytes)

```
AI = (2 × 234,752) / (939,008 + 4,712)
   = 469,504 / 943,720
   = 0.497 FLOPs/byte
```
