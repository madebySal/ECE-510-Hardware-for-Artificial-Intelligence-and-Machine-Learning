"""Generate CF09 roofline plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os

out_dir = os.path.dirname(__file__)

# ── Platform constants ──────────────────────────────────────────────────────
PEAK_COMPUTE_GOPS  = 12.8          # 128 ops/cycle × 100 MHz
PEAK_BW_GBs        = 1.5625e-3     # SPI 12.5 MHz → 1.5625 MB/s = 1.5625e-3 GB/s
RIDGE_AI           = PEAK_COMPUTE_GOPS / PEAK_BW_GBs   # 8192 FLOPs/byte

AI_LOW   = 8.0    # no activation reuse
AI_HIGH  = 16.0   # perfect activation reuse

ATTAIN_LOW  = AI_LOW  * PEAK_BW_GBs   # GOPS  (= 0.0125)
ATTAIN_HIGH = AI_HIGH * PEAK_BW_GBs   # GOPS  (= 0.025)

MEASURED_GOPS = 8.88e-3  # from co-sim: 128 ops / 14.41 µs


# ════════════════════════════════════════════════════════════════════════════
# Plot 1 — CMAN hand-drawn style roofline sketch
# ════════════════════════════════════════════════════════════════════════════
def cman_sketch():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#fffef5')
    ax.set_facecolor('#fffef5')

    ai_range = np.logspace(-1, 4, 500)

    # BW ceiling (memory-bound slope)
    bw_line = PEAK_BW_GBs * ai_range
    # Compute ceiling (flat)
    compute_line = np.full_like(ai_range, PEAK_COMPUTE_GOPS)
    # Roofline = min of the two
    roof = np.minimum(bw_line, compute_line)

    ax.loglog(ai_range, roof, 'k-', lw=2.2, label='Roofline (sky130A, SPI interface)')

    # Annotate BW ceiling slope
    ax.text(0.25, PEAK_BW_GBs * 0.2, f'BW ceiling\n(slope = {PEAK_BW_GBs*1e3:.4f} GB/s)',
            fontsize=8, color='#555', rotation=45, ha='left')

    # Annotate compute ceiling
    ax.axhline(PEAK_COMPUTE_GOPS, color='#888', ls='--', lw=1.2)
    ax.text(2000, PEAK_COMPUTE_GOPS * 1.35,
            f'Compute ceiling\n(peak = {PEAK_COMPUTE_GOPS} GOPS @ 100 MHz)',
            fontsize=8, color='#555', ha='center')

    # Ridge point
    ax.plot(RIDGE_AI, PEAK_COMPUTE_GOPS, 'k^', ms=9, zorder=5)
    ax.annotate(f'Ridge point\n({RIDGE_AI:.0f} FLOPs/byte,\n{PEAK_COMPUTE_GOPS} GOPS)',
                xy=(RIDGE_AI, PEAK_COMPUTE_GOPS),
                xytext=(RIDGE_AI * 0.3, PEAK_COMPUTE_GOPS * 2.0),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='k'),
                ha='center')

    # Kernel AI lower bound
    ax.plot(AI_LOW, ATTAIN_LOW, 'rv', ms=11, zorder=6, label=f'Kernel AI lower = {AI_LOW} FLOPs/byte')
    ax.annotate(f'Kernel lower bound\nAI = {AI_LOW} FLOPs/byte\n{ATTAIN_LOW*1e3:.1f} MOPS attainable',
                xy=(AI_LOW, ATTAIN_LOW),
                xytext=(AI_LOW * 3.5, ATTAIN_LOW * 0.2),
                fontsize=8.5, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'),
                ha='center')

    # Kernel AI upper bound
    ax.plot(AI_HIGH, ATTAIN_HIGH, 'bs', ms=11, zorder=6, label=f'Kernel AI upper = {AI_HIGH} FLOPs/byte')
    ax.annotate(f'Kernel upper bound\nAI = {AI_HIGH} FLOPs/byte\n{ATTAIN_HIGH*1e3:.1f} MOPS attainable',
                xy=(AI_HIGH, ATTAIN_HIGH),
                xytext=(AI_HIGH * 3.5, ATTAIN_HIGH * 4),
                fontsize=8.5, color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue'),
                ha='center')

    # Shade memory-bound region
    ax.axvspan(0.1, RIDGE_AI, alpha=0.07, color='orange', label='Memory-bound region')
    ax.axvspan(RIDGE_AI, 1e4, alpha=0.07, color='steelblue', label='Compute-bound region')

    ax.set_xlabel('Arithmetic Intensity (binary FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Attainable Performance (GOPS)', fontsize=12)
    ax.set_title('CMAN Roofline Sketch — BNN Accelerator (N=64)\n'
                 'sky130A / sky130_fd_sc_hd, 100 MHz, SPI interface @ 12.5 MHz',
                 fontsize=11)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e-4, 1e2)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Note on binary ops
    ax.text(0.01, 0.02,
            '1 binary op (XNOR/add/compare) = 1 "FLOP" for this metric',
            transform=ax.transAxes, fontsize=7.5, color='#666', style='italic')

    plt.tight_layout()
    path = os.path.join(out_dir, 'cman_roofline_sketch.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Plot 2 — CLLM benchmarking roofline (accelerator point on roofline)
# ════════════════════════════════════════════════════════════════════════════
def cllm_roofline():
    # SW baseline platform specs (Intel i7-13700, single-threaded NumPy)
    SW_PEAK_GFLOPS  = 4.10   # measured: 469,504 FLOPs / 114.40 µs
    SW_PEAK_BW_GBs  = 50.0   # DDR5 @ 4400 MT/s ×2 channels, 64 bit = ~70 GB/s peak;
                             # practical single-thread BW ~50 GB/s
    SW_RIDGE        = SW_PEAK_GFLOPS / SW_PEAK_BW_GBs  # ~0.082 FLOPs/byte

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('#f8f8ff')
    ax.set_facecolor('#f8f8ff')

    ai_range = np.logspace(-2, 5, 600)

    # sky130 roofline
    sky130_roof = np.minimum(PEAK_BW_GBs * ai_range, PEAK_COMPUTE_GOPS)
    ax.loglog(ai_range, sky130_roof, 'k-', lw=2.5, label='sky130A ASIC roofline (SPI BW)')

    # CPU roofline (software baseline platform)
    cpu_roof = np.minimum(SW_PEAK_BW_GBs * ai_range, SW_PEAK_GFLOPS)
    ax.loglog(ai_range, cpu_roof, 'b--', lw=2.0, alpha=0.7,
              label='i7-13700 CPU roofline (M1 baseline platform)')

    # CPU ridge point
    ax.plot(SW_RIDGE, SW_PEAK_GFLOPS, 'b^', ms=8, zorder=5)
    ax.text(SW_RIDGE * 0.4, SW_PEAK_GFLOPS * 1.5,
            f'CPU ridge\n({SW_RIDGE:.3f} FLOPs/B)', fontsize=7.5, color='blue', ha='center')

    # sky130 ridge point
    ax.plot(RIDGE_AI, PEAK_COMPUTE_GOPS, 'k^', ms=9, zorder=5)
    ax.text(RIDGE_AI * 0.3, PEAK_COMPUTE_GOPS * 1.6,
            f'ASIC ridge\n({RIDGE_AI:.0f} FLOPs/B)', fontsize=7.5, ha='center')

    # SW baseline measured operating point
    # AI = 8 FLOPs/byte (no-reuse; CPU doesn't cache weights between inferences)
    # SW throughput = 4.10 GFLOPS (measured)
    SW_AI = AI_LOW  # same kernel, AI = 8
    ax.plot(SW_AI, SW_PEAK_GFLOPS, 'b*', ms=14, zorder=7,
            label=f'SW baseline (measured): {SW_PEAK_GFLOPS:.2f} GOPS @ AI={SW_AI}')
    ax.annotate(f'M1 SW baseline\n(measured)\n{SW_PEAK_GFLOPS:.2f} GOPS',
                xy=(SW_AI, SW_PEAK_GFLOPS),
                xytext=(SW_AI * 8, SW_PEAK_GFLOPS * 0.4),
                fontsize=8, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # HW accelerator — measured from co-simulation
    HW_AI_MEASURED = AI_LOW  # no activation reuse in per-neuron co-sim test
    ax.plot(HW_AI_MEASURED, MEASURED_GOPS, 'r*', ms=14, zorder=7,
            label=f'HW accel (measured, co-sim): {MEASURED_GOPS*1e3:.1f} MOPS @ AI={HW_AI_MEASURED}')
    ax.annotate(f'HW accelerator\n(measured, co-sim)\n{MEASURED_GOPS*1e3:.1f} MOPS',
                xy=(HW_AI_MEASURED, MEASURED_GOPS),
                xytext=(HW_AI_MEASURED * 0.2, MEASURED_GOPS * 0.08),
                fontsize=8, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'),
                ha='center')

    # HW accelerator — upper bound (perfect act reuse)
    ax.plot(AI_HIGH, ATTAIN_HIGH, 'rs', ms=10, zorder=6, alpha=0.7,
            label=f'HW accel (upper bound): {ATTAIN_HIGH*1e3:.1f} MOPS @ AI={AI_HIGH}')

    ax.set_xlabel('Arithmetic Intensity (binary FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Attainable Performance (GOPS)', fontsize=12)
    ax.set_title('CLLM Roofline — BNN Accelerator vs. M1 Software Baseline\n'
                 'sky130A ASIC (N=64, co-simulation) vs. Intel i7-13700 (NumPy)',
                 fontsize=11)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-5, 1e2)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.35)

    plt.tight_layout()
    path = os.path.join(out_dir, 'benchmarks', 'roofline_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    cman_sketch()
    cllm_roofline()
    print("Done.")
