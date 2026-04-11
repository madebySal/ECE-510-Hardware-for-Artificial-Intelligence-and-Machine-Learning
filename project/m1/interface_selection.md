# Interface Selection — BNN Inference Accelerator
## ECE 510 Spring 2026 — project/m1/interface_selection.md

---

## Selected interface: AXI4-Lite (control) + AXI4-Stream (data)

---

## Host platform

The assumed host is an **FPGA SoC** (e.g., Xilinx Zynq or similar ARM + FPGA
fabric), where the ARM CPU handles pre/post-processing and the programmable
logic implements the BNN accelerator chiplet. AXI4 is the native interface bus
for Zynq SoC designs and is directly supported by OpenLane 2 flows targeting
FPGA-adjacent synthesis.

---

## Bandwidth requirement at the target operating point

Target throughput: **1,000,000 inferences/sec** (aggressive edge case for
keyword spotting at high sample rates).

Per-inference data transferred across the host–chiplet interface:
- Input activation vector: 784 bits = **98 bytes**
- Output class scores: 10 × FP32 = **40 bytes**
- Weights: loaded once at startup and held in on-chip SRAM; **not transferred
  per inference**

```
Required BW = (98 + 40) bytes × 1,000,000 inferences/s
            = 138,000,000 bytes/s
            = 138 MB/s
            ≈ 0.138 GB/s
```

At the baseline target of **1,000 inferences/sec** (more realistic for keyword
spotting):
```
Required BW = 138 bytes × 1,000 = 138,000 bytes/s ≈ 0.14 MB/s
```

---

## Interface rated bandwidth vs. required bandwidth

| Interface | Rated bandwidth | Required (1K inf/s) | Required (1M inf/s) | Bottleneck? |
|---|---|---|---|---|
| SPI (50 Mbit/s) | 6.25 MB/s | 0.14 MB/s | 138 MB/s | No at 1K; Yes at 1M |
| AXI4-Lite (32-bit, 100 MHz) | ~400 MB/s | 0.14 MB/s | 138 MB/s | No at either point |
| AXI4-Stream (32-bit, 100 MHz) | ~400 MB/s | 0.14 MB/s | 138 MB/s | No at either point |

**The design is not interface-bound** at either operating point when using
AXI4-Lite + AXI4-Stream. The interface bandwidth (≈400 MB/s) is **2,900×
the required bandwidth** at 1K inf/s and **2.9× at 1M inf/s**.

The roofline analysis confirms this: the kernel's arithmetic intensity is
15.11 FLOP/byte, placing it in the memory-bound regime relative to SRAM
bandwidth — not interface bandwidth. The host interface is not the bottleneck.

---

## Justification for AXI4-Lite + AXI4-Stream over alternatives

**Why not SPI:** SPI tops out at ~6.25 MB/s. At 1M inferences/sec the required
138 MB/s exceeds SPI capacity by 22×. SPI is appropriate for MCU-class hosts
with very low inference rates only.

**Why not I²C:** Maximum 3.4 Mbit/s = 0.43 MB/s — far below even the moderate
1K inf/s case. Ruled out immediately.

**Why not PCIe:** PCIe adds significant implementation complexity (endpoint
controller IP, root complex on host side) that is not justified for an FPGA SoC
target. PCIe is appropriate for data-center accelerators, not edge inference
chiplets.

**Why AXI4-Lite + AXI4-Stream:**
- AXI4-Lite handles control-plane traffic: writing configuration registers
  (layer dimensions, start/stop), reading status registers (inference complete,
  output ready). Simple register-mapped interface, well-supported in OpenLane 2.
- AXI4-Stream handles data-plane traffic: streaming the 98-byte input activation
  vector in and the 40-byte output vector out. Unidirectional, no address
  overhead, natural fit for inference pipelines with fixed-size tensors.
- Both are ARM AMBA standards with extensive open-source IP and synthesis
  support. This combination is the standard choice for FPGA SoC accelerator
  integration and is what FINN (the reference BNN accelerator framework) uses.
