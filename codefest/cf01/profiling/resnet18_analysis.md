# ResNet18 Profiling Analysis
## ECE 510 – CF01 | Spring 2026

**Platform:** Intel Core i7-1165G7 @ 2.80GHz, 16 GB DDR4, PyTorch 2.2.0  
**Input:** batch_size=1, 3×224×224 RGB image, FP32  

---

## Dominant Kernel

cProfile identifies `conv2d` as the dominant kernel, consuming **80.1%** of total inference
runtime (3.871 s out of 4.832 s across 20 runs). All subsequent analysis targets the
convolutional layers as the hardware acceleration candidate.

---

## Top-5 Layers by MAC Count

MACs computed analytically from architecture dimensions (Cout × Cin × Kh × Kw × Hout × Wout),
consistent with torchinfo / fvcore output for ResNet-18 (batch=1, 3×224×224, FP32).

| Rank | Layer | Output Shape | Kernel | MACs (M) | Params | % of Total MACs |
|------|-------|-------------|--------|----------|--------|-----------------|
| 1 | `conv1` | 64×112×112 | 7×7 | 118.01 | 9,408 | 6.5% |
| 2 | `layer1.0.conv1` | 64×56×56 | 3×3 | 115.61 | 36,864 | 6.4% |
| 3 | `layer1.0.conv2` | 64×56×56 | 3×3 | 115.61 | 36,864 | 6.4% |
| 4 | `layer1.1.conv1` | 64×56×56 | 3×3 | 115.61 | 36,864 | 6.4% |
| 5 | `layer1.1.conv2` | 64×56×56 | 3×3 | 115.61 | 36,864 | 6.4% |
| – | **Total (all layers)** | – | – | **~1,813.6** | – | **100%** |

> Each MAC = 1 multiply + 1 add = 2 FLOPs.  
> Total FLOPs ≈ 3.627 GFLOP per image.

---

## Arithmetic Intensity — Most MAC-Intensive Layer (`conv1`)

`conv1` is the single most MAC-intensive layer (118.01 M MACs, 7×7 kernel, 64 output channels).

Arithmetic intensity (AI) = FLOPs / bytes of memory traffic, assuming all weights and
activations are loaded from DRAM with no reuse.

### FLOPs

```
MACs  = Cout × Cin × Kh × Kw × Hout × Wout
      = 64 × 3 × 7 × 7 × 112 × 112
      = 118,013,952

FLOPs = 2 × MACs = 236,027,904
```

### Memory Traffic (FP32, no reuse)

| Tensor | Dimensions | Bytes |
|--------|-----------|-------|
| Weights | 64 × 3 × 7 × 7 × 4 B | 37,632 |
| Input activation | 3 × 224 × 224 × 4 B | 602,112 |
| Output activation | 64 × 112 × 112 × 4 B | 3,211,264 |
| **Total** | | **3,851,008** |

### Result

```
AI = 236,027,904 FLOPs / 3,851,008 bytes = 61.29 FLOPs/byte
```

---

## Roofline Placement

**CPU specs (Intel i7-1165G7):**
- Peak compute: ~150 GFLOP/s (FP32, AVX2)
- Peak memory bandwidth: ~40 GB/s (DDR4 dual-channel)
- Ridge point: 150 / 40 = **3.75 FLOPs/byte**

**Result:** AI = 61.29 FLOPs/byte >> ridge point of 3.75 FLOPs/byte  
→ `conv1` is firmly **compute-bound** on this CPU.

The theoretical ceiling at AI=61.29 is ~150 GFLOP/s (capped by compute roof),
but the achieved rate is only ~14.9 GFLOP/s — roughly **10× below the roofline ceiling** —
indicating that a custom hardware accelerator with better data reuse and parallelism
has significant headroom to improve performance.

---

## HW/SW Partition Proposal

| Component | Target | Rationale |
|-----------|--------|-----------| 
| Conv2d (XNOR-popcount) | **Hardware (chiplet)** | Dominant kernel, compute-bound, parallelizable |
| BatchNorm | Software (host CPU) | Low FLOPs, parameter-dependent, simpler in SW |
| ReLU / binarize | Hardware (fused) | Trivial logic, fuse with conv output |
| FC layer | Software | Small, infrequent, not bottleneck |

---

## Interface Selection

**Selected interface: SPI**

The BNN chiplet targets edge deployment on an MCU-class host (ARM Cortex-M class).
Per-sample input data is 784/8 = 98 bytes; weight memory is (784×256 + 256×10)/8 ≈ 25,920 bytes.
At SPI's ~50 Mbit/s (~6.25 MB/s), a full weight load transfers in ~4 ms and each input
in <0.2 ms. Given that BNN hardware inference takes <1 ms, SPI does not become the
bottleneck for single-sample keyword-spotting or gesture-recognition workloads. The
design is therefore **not interface-bound** at this operating point. AXI4 would offer
higher bandwidth but adds implementation complexity unjustified at this data scale.
