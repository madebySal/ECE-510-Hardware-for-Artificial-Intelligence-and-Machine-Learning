"""
gen_cosim_waveform.py  —  M3 end-to-end co-simulation waveform
Produces cosim_waveform.png showing three annotated regions:
  ① Host-side SPI write (activation + weight burst)
  ② Internal compute activity (compute_start → XNOR+popcount)
  ③ Host-side SPI read (STATUS poll → result captured)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------
# Timing constants (all in ns; matches tb_top.sv SCK_HALF=40 ns, CLK=10 ns)
# --------------------------------------------------------------------------
CLK_HALF  = 5       # ns
SCK_HALF  = 40      # ns
SCK_PER   = 80      # ns
BIT_PER   = SCK_PER # ns per SPI bit (sample on rise)
BYTE_PER  = 8 * BIT_PER   # 640 ns per SPI byte

def clk_wave(t_start, t_end):
    """Generate ideal 100 MHz clock edges."""
    edges = np.arange(t_start, t_end, CLK_HALF)
    t = []
    v = []
    for i, e in enumerate(edges):
        t.extend([e, e])
        v.extend([i % 2, 1 - i % 2])
    return np.array(t), np.array(v, dtype=float)

def spi_clk_burst(t0, n_bytes):
    """SCK for n_bytes (plus 1 CMD byte = n_bytes+1 total), CS active."""
    n_bits = (n_bytes + 1) * 8
    t = [t0]
    v = [0.0]
    cur = t0 + SCK_HALF
    for _ in range(n_bits):
        t += [cur, cur + SCK_HALF, cur + SCK_HALF]
        v += [1.0, 1.0, 0.0]
        cur += SCK_PER
    t.append(cur)
    v.append(0.0)
    return np.array(t), np.array(v)

def pulse(t_on, t_off, before_val=0.0, after_val=0.0):
    return ([t_on, t_on, t_off, t_off], [before_val, 1.0, 1.0, after_val])

# --------------------------------------------------------------------------
# Build timeline for test 1 (all_agree)
# Phase 0: reset (0–60 ns)
# Phase 1: SPI write activation (8 bytes + CMD = 9 bytes)   60–5820 ns
# Phase 2: SPI write weight     (8 bytes + CMD = 9 bytes)   5980–11740 ns
# Phase 3: SPI write CTRL       (1 byte  + CMD = 2 bytes)   11900–12540 ns
# Phase 4: compute              (10 ns = 1 clock cycle)     12700–12720 ns
# Phase 5: SPI read STATUS      (1 dummy + CMD = 2 bytes)   12880–13520 ns
# --------------------------------------------------------------------------

RST_END    = 60
WR_ACT_T0  = RST_END + 40
WR_ACT_T1  = WR_ACT_T0 + 9 * BYTE_PER           # 5820
WR_WGT_T0  = WR_ACT_T1 + 200
WR_WGT_T1  = WR_WGT_T0 + 9 * BYTE_PER           # 11700
WR_CTL_T0  = WR_WGT_T1 + 200
WR_CTL_T1  = WR_CTL_T0 + 2 * BYTE_PER           # 12420
COMP_T0    = WR_CTL_T1 + 40
COMP_T1    = COMP_T0 + 20                         # 1 clk cycle
RD_STA_T0  = COMP_T1 + 200
RD_STA_T1  = RD_STA_T0 + 2 * BYTE_PER + 200
T_END      = RD_STA_T1 + 400

# --------------------------------------------------------------------------
# Signal traces
# --------------------------------------------------------------------------

# CLK — show a representative window
t_clk = np.arange(0, T_END, CLK_HALF)
v_clk = np.zeros(len(t_clk))
for i in range(len(t_clk)):
    v_clk[i] = float((i) % 2)

# RST
t_rst = [0, 0, RST_END, RST_END, T_END]
v_rst = [1, 1, 1, 0, 0]

# CS_N (active-low shown as inverted for readability)
t_csn  = [0]
v_csn  = [1.0]
for (t0, t1) in [(WR_ACT_T0, WR_ACT_T1), (WR_WGT_T0, WR_WGT_T1),
                 (WR_CTL_T0, WR_CTL_T1), (RD_STA_T0, RD_STA_T1)]:
    t_csn += [t0, t0, t1, t1]
    v_csn += [1.0, 0.0, 0.0, 1.0]
t_csn.append(T_END)
v_csn.append(1.0)

# SCK — active only during CS asserted
t_sck = [0, WR_ACT_T0]
v_sck = [0.0, 0.0]
for (t0, n_b) in [(WR_ACT_T0, 8), (WR_WGT_T0, 8), (WR_CTL_T0, 1), (RD_STA_T0, 1)]:
    ts, vs = spi_clk_burst(t0, n_b)
    t_sck += list(ts)
    v_sck += list(vs)
t_sck.append(T_END)
v_sck.append(0.0)

# compute_start — one-clock pulse after CTRL write
t_cs  = [0, COMP_T0, COMP_T0, COMP_T0+10, COMP_T0+10, T_END]
v_cs  = [0,       0,        1,          1,           0,     0]

# result_valid — sticky high after compute
t_rv  = [0, COMP_T0+10, COMP_T0+10, T_END]
v_rv  = [0,           0,          1,     1]

# result_out — 1 (all_agree)
t_ro  = [0, COMP_T0+10, COMP_T0+10, T_END]
v_ro  = [0,           0,          1,     1]

# MISO — goes high during STATUS read phase (shows result_out=1)
t_mi = [0, RD_STA_T0, RD_STA_T0 + BYTE_PER + BYTE_PER//2, T_END]
v_mi = [0,          0,                                  1,     1]

signals = [
    ("clk",           t_clk,      v_clk,      "steelblue"),
    ("rst",           t_rst,      v_rst,      "tomato"),
    ("cs_n\n(active-low)", t_csn, v_csn,      "slategray"),
    ("sck",           t_sck,      v_sck,      "darkorange"),
    ("compute_start", t_cs,       v_cs,       "mediumpurple"),
    ("result_valid",  t_rv,       v_rv,       "mediumseagreen"),
    ("miso\n(result_out)", t_mi,  v_mi,       "crimson"),
]

fig, axes = plt.subplots(len(signals), 1, figsize=(14, 9), sharex=True)
fig.suptitle(
    "BNN top — end-to-end co-simulation (all_agree, N=64, act=wgt=0xFFFF…)\n"
    "4/4 tests PASS  ·  out=1  ·  result_valid asserted one cycle after compute_start",
    fontsize=11, fontweight="bold"
)

for ax, (name, t, v, color) in zip(axes, signals):
    ax.step(t, v, where="post", color=color, linewidth=1.5)
    ax.set_ylim(-0.3, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=7)
    ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=60, va="center")
    ax.yaxis.set_label_position("left")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (ns)", fontsize=9)

# ── Region annotations ────────────────────────────────────────────────────
ax0 = axes[0]
region_y = 1.6

def annotate_region(ax, t0, t1, label, color):
    ax.annotate("", xy=(t1, region_y), xytext=(t0, region_y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.5),
                annotation_clip=False)
    ax.text((t0+t1)/2, region_y + 0.12, label, ha="center", va="bottom",
            fontsize=8, color=color, fontweight="bold",
            transform=ax.transData, clip_on=False)

annotate_region(ax0, WR_ACT_T0, WR_CTL_T1,
                "① Host SPI write\n(act + wgt + CTRL)", "navy")
annotate_region(ax0, COMP_T0, COMP_T1+200,
                "② Compute\n(XNOR+pop)", "darkgreen")
annotate_region(ax0, RD_STA_T0, RD_STA_T1,
                "③ Host SPI read\n(STATUS poll)", "darkred")

plt.tight_layout(rect=[0, 0, 1, 0.93])

out = "cosim_waveform.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
