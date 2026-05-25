# CMAN — AER Bandwidth Analysis
**ECE 410/510 — Codefest 08**  
**N = 1024 neurons, f = 50 Hz mean firing rate**

---

## Task 1 — Mean Aggregate Spike Rate R

Formula: R = N × f

```
R = 1024 × 50 = 51,200 spikes/second
```

---

## Task 2 — Mean AER Bandwidth B

Each AER packet = 20 bits total:
- 10-bit neuron address (log₂(1024) = 10)
- 6-bit timestamp
- 4-bit framing/parity overhead

Formula: B = R × 20

```
B = 51,200 × 20 = 1,024,000 bits/second = 1.024 Mbit/s
```

---

## Task 3 — Interface Comparison

| Interface | Max Bandwidth | Sustains 1.024 Mbit/s? | Notes |
|-----------|--------------|------------------------|-------|
| SPI | ≤50 Mbit/s | ✅ Yes | 48.8× headroom — overkill |
| I²C | ≤3.4 Mbit/s | ✅ Yes | 3.3× headroom — sufficient |
| AXI4-Lite | ~100 Mbit/s | ✅ Yes | Far exceeds requirement |

**Lowest-complexity interface that suffices: I²C**

I²C handles 1.024 Mbit/s comfortably with 3.3× headroom at mean firing rate.
No need for SPI or AXI4-Lite at mean rate alone.

---

## Task 4 — Burst Bandwidth Analysis

25% of 1024 neurons fire within a 1 ms window:

```
Burst spikes = 0.25 × 1024 = 256 spikes in 1 ms
Burst rate   = 256 / 0.001 = 256,000 spikes/second
Peak BW      = 256,000 × 20 = 5,120,000 bits/s = 5.12 Mbit/s
```

**Burst-to-mean ratio:**
```
5.12 Mbit/s / 1.024 Mbit/s = 5×
```

**Can I²C absorb the burst?**

I²C max = 3.4 Mbit/s < 5.12 Mbit/s — **No. I²C cannot absorb the burst.**

Buffering is required. Excess overflow during 1 ms burst:
```
Excess rate  = 5.12 - 3.4 = 1.72 Mbit/s
Bytes needed = (1.72 × 10⁶ × 0.001) / 8 ≈ 215 bytes
```

**Decision: A 256-byte FIFO buffer is sufficient to absorb burst overflow.**

---

## Task 5 — Frame-Based Comparison

Frame-based readout samples all 1024 neurons every 1 ms, 1 bit per neuron:

```
Frame BW = 1024 bits / 0.001s = 1,024,000 bits/s = 1.024 Mbit/s
```

**AER-to-frame ratio at f = 50 Hz:**
```
AER BW   = 1.024 Mbit/s
Frame BW = 1.024 Mbit/s
Ratio    = 1.024 / 1.024 = 1×
```

**Crossover firing rate f_crossover:**

Set AER bandwidth = Frame bandwidth and solve for f:
```
N × f × 20 = N × 1000
f × 20 = 1000
f_crossover = 50 Hz
```

**AER-to-frame ratio at mean firing rate = 1× (they are equal at f = 50 Hz)**

**One-sentence implication:**
AER is the right choice when the mean firing rate is below 50 Hz — at sparser
activity, AER transmits far less data than frame-based readout, making it
bandwidth-efficient for low-activity neuromorphic systems.
