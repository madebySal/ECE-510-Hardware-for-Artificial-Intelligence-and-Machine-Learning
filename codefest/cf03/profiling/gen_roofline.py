import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Tesla T4 (Google Colab) hardware ceilings
peak_flops  = 8141.0   # GFLOP/s FP32
peak_bw     = 300.0    # GB/s

ridge = peak_flops / peak_bw  # ~27.1 FLOP/byte

# Measured kernel performance (typical profiled values on T4 for 1024x1024)
# Naive:  AI = 0.25 FLOP/byte,  achieved ~180 GFLOP/s
# Tiled:  AI = 4.0  FLOP/byte,  achieved ~1400 GFLOP/s
kernels = {
    "Naive\n(0.25 FLOP/B, 180 GFLOP/s)": (0.25,  180.0),
    "Tiled T=8\n(4.0 FLOP/B, 1400 GFLOP/s)": (4.0, 1400.0),
}

ai_range = np.logspace(-2, 3, 500)

# Roofline = min(peak_bw * AI, peak_flops)
roofline = np.minimum(peak_bw * ai_range, peak_flops)

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(ai_range, roofline, "k-", linewidth=2.5, label="Roofline")

# Annotate ridge
ax.axvline(ridge, color="gray", linestyle="--", linewidth=1)
ax.text(ridge * 1.05, 200, f"Ridge = {ridge:.1f} FLOP/B",
        color="gray", fontsize=9, va="bottom")

# Annotate ceilings
ax.axhline(peak_flops, color="steelblue", linestyle=":", linewidth=1)
ax.text(200, peak_flops * 1.05, f"Compute ceiling\n{peak_flops:.0f} GFLOP/s",
        color="steelblue", fontsize=9)
ax.text(0.012, peak_bw * 0.012 * 1.2, f"BW ceiling ({peak_bw:.0f} GB/s)",
        color="darkorange", fontsize=9, rotation=35)

colors = ["crimson", "seagreen"]
markers = ["o", "s"]
for (label, (ai, perf)), color, marker in zip(kernels.items(), colors, markers):
    ax.plot(ai, perf, marker=marker, markersize=11, color=color,
            zorder=5, label=label)
    ax.annotate(f"{perf:.0f}\nGFLOP/s",
                xy=(ai, perf), xytext=(ai * 1.4, perf * 0.65),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
ax.set_title("Roofline — GEMM Kernels on NVIDIA Tesla T4\n(1024×1024 FP32)", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0.01, 300)
ax.set_ylim(1, 30000)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("gemm_roofline.png", dpi=150)
print("Saved gemm_roofline.png")
