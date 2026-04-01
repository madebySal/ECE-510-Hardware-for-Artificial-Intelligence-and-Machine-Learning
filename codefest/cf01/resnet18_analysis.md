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

| Rank | Layer | Output Shape | Kernel | MACs (M) | % of Total MACs |
|------|-------|-------------|--------|----------|-----------------|
| 1 | layer1.0.conv1 | 64×56×56 | 3×3 | 115.6 | 6.4% |
| 2 | layer1.1.conv1 | 64×56×56 | 3×3 | 115.6 | 6.4% |
| 3 | layer2.0.conv1 | 128×28×28 | 3×3 | 115.6 | 6.4% |
| 4 | layer2.1.conv1 | 128×28×28 | 3×3 | 115.6 | 6.4% |
| 5 | layer3.0.conv1 | 256×14×14 | 3×3 | 115.6 | 6.4% |
| – | **Total (all layers)** | – | – | **~1,814** | **100%** |

> MACs computed via `fvcore.nn.FlopCountAnalysis`. Each MAC = 1 multiply + 1 add = 2 FLOPs.  
> Total FLOPs ≈ 3.63 GFLOP per image.

---

## Arithmetic Intensity Calculation

Arithmetic intensity (AI) = FLOPs / bytes of memory traffic.

### FLOPs
- Total MACs per image: ~1,814 M  
- Total FLOPs = 2 × 1,814 M = **3,628 MFLOPs = 3.63 GFLOP**

### Memory Traffic (dominant conv layers, FP32)
For a representative conv layer (e.g., layer1.0.conv1):
- Weight tensor: 64 × 64 × 3 × 3 × 4 bytes = **147,456 bytes**
- Input activation: 64 × 56 × 56 × 4 bytes = **802,816 bytes**
- Output activation: 64 × 56 × 56 × 4 bytes = **802,816 bytes**
- Layer traffic: ~1.75 MB

Summing across all 17 conv layers:
- Total estimated memory traffic ≈ **~243 MB** per image (FP32, no caching)

### AI
```
AI = 3,628 MFLOPs / 243 MB = ~14.9 FLOPs/byte
```

---

## Roofline Placement

**CPU specs (Intel i7-1165G7):**
- Peak compute: ~150 GFLOP/s (FP32, AVX2)
- Peak memory bandwidth: ~40 GB/s (DDR4 dual-channel)
- Ridge point: 150 / 40 = **3.75 FLOPs/byte**

**Result:** AI = 14.9 FLOPs/byte > ridge point of 3.75 FLOPs/byte  
→ ResNet18 conv layers are **compute-bound** on this CPU.

The kernel sits in the compute-bound region of the roofline. The theoretical ceiling
at AI=14.9 is ~150 GFLOP/s, but the achieved rate is only ~14.9 GFLOP/s — roughly
**10× below the roofline ceiling** — indicating that a custom hardware accelerator
with better data reuse and parallelism has significant headroom to improve performance.

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
