# Interface Selection – BNN Inference Accelerator
## ECE 510 – Milestone 1 | Spring 2026

---

## Selected Interface: SPI

**Interface chosen:** SPI (Serial Peripheral Interface)  
**Host platform assumed:** ARM Cortex-M class MCU (edge deployment, e.g., STM32 or nRF52840)

---

## Bandwidth Requirement Calculation

**Target operating point:** 10,000 inferences/sec (≈10× software baseline throughput)

**Data transferred per inference (1-bit packed binary):**

| Operand | Size | Calculation |
|---------|------|-------------|
| Input activation (binary, packed) | 98 bytes | 784 bits / 8 = 98 bytes |
| Output logits (FP32) | 40 bytes | 10 × 4 bytes |
| **Total per inference** | **138 bytes** | |

**Required bandwidth:**
```
BW_required = 138 bytes × 10,000 inferences/sec = 1,380,000 bytes/sec ≈ 1.38 MB/s
```

**SPI rated bandwidth:** 50 Mbit/s = 6.25 MB/s

**Comparison:**

| | Value |
|--|-------|
| Required bandwidth | 1.38 MB/s |
| SPI rated bandwidth | 6.25 MB/s |
| Headroom margin | 4.5× |

---

## Bottleneck Status

The design is **not interface-bound** at the 10,000 inferences/sec target. SPI provides 4.5×
more bandwidth than required. The interface does not appear on the roofline as a bottleneck.

**Note:** Weights (~31 KB total for all three layers, 1-bit packed) are loaded once at startup
over SPI (31,232 bytes / 6.25 MB/s ≈ 5 ms one-time load) and cached in on-chip SRAM.
Subsequent inferences only transfer the 138-byte input/output payload per sample.

---

## Justification for SPI over Alternatives

| Interface | Bandwidth | Verdict |
|-----------|-----------|---------|
| SPI (50 Mbit/s) | 6.25 MB/s | **Selected** — sufficient, simple, widely supported on MCUs |
| I²C (400 kHz) | 0.05 MB/s | Insufficient — below required 1.38 MB/s |
| AXI4-Lite | ~GB/s | Overkill for MCU host; adds complexity |
| PCIe / UCIe | ~GB/s | Data-center target only; not applicable |

SPI is the correct choice for an edge MCU host at this data scale. AXI4 would be appropriate
only if the host were an FPGA SoC or if the target throughput scaled to >100,000 inferences/sec.
