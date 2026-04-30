"""Generate waveform.png for M2 submission from compute_core test vector."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Signals from tb_compute_core test: all_agree case (N=64, act=all-ones, wgt=all-ones)
# CLK period=10ns. Reset released at t=10. Inputs applied t=20. Result registered at t=30.
# Times in ns
events = [
    # t,  clk, rst, act_valid, wgt_valid, data_ready, result_valid, out
    (  0,   0,   1,   0,  0,  0,  0,  0),
    (  5,   1,   1,   0,  0,  0,  0,  0),
    ( 10,   0,   1,   0,  0,  0,  0,  0),
    ( 15,   1,   0,   0,  0,  0,  0,  0),  # rst released
    ( 20,   0,   0,   1,  1,  1,  0,  0),  # inputs applied
    ( 25,   1,   0,   1,  1,  1,  0,  0),  # rising edge: compute
    ( 30,   0,   0,   1,  1,  1,  1,  1),  # result registered
    ( 35,   1,   0,   1,  1,  1,  1,  1),
    ( 40,   0,   0,   1,  1,  1,  1,  1),
    ( 45,   1,   0,   1,  1,  1,  1,  1),
    ( 50,   0,   0,   0,  0,  0,  1,  1),  # inputs released
    ( 55,   1,   0,   0,  0,  0,  1,  1),
    ( 60,   0,   0,   0,  0,  0,  1,  1),
]

labels  = ["clk", "rst", "act_valid", "wgt_valid", "data_ready", "result_valid", "out"]
cols    = [1, 2, 3, 4, 5, 6, 7]
colors  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#8BC34A"]
titles  = ["clk", "rst", "act_valid", "wgt_valid", "data_ready\n(act&wgt)", "result_valid", "out"]

times = [e[0] for e in events]

def step_xy(ts, vs):
    xs, ys = [ts[0]], [vs[0]]
    for i in range(1, len(ts)):
        xs.append(ts[i]); ys.append(vs[i-1])
        xs.append(ts[i]); ys.append(vs[i])
    return xs, ys

fig, axes = plt.subplots(len(labels), 1, figsize=(12, 8), sharex=True)
fig.suptitle("compute_core — all_agree test vector (N=64, act=wgt=0xFFFF…)\n"
             "pop=64 > 32 → out=1, result_valid asserted one cycle after data_ready",
             fontsize=11, fontweight="bold")

for ax, label, col, color, title in zip(axes, labels, cols, colors, titles):
    vals = [e[col] for e in events]
    xs, ys = step_xy(times, vals)
    ax.plot(xs, ys, color=color, linewidth=1.8)
    ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=60, va="center")
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=7)
    ax.set_xlim(times[0], times[-1] + 3)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

# Annotate key events
axes[0].axvline(15, color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(25, color="green", linestyle="--", alpha=0.5)
axes[0].axvline(30, color="blue",  linestyle="--", alpha=0.5)
axes[-1].set_xlabel("Time (ns)", fontsize=10)

fig.text(0.18, 0.02, "rst released\n@15 ns",  ha="center", fontsize=7, color="gray")
fig.text(0.40, 0.02, "compute\nedge @25 ns",  ha="center", fontsize=7, color="green")
fig.text(0.52, 0.02, "result valid\n@30 ns",   ha="center", fontsize=7, color="blue")

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = os.path.join(os.path.dirname(__file__), "sim", "waveform.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
