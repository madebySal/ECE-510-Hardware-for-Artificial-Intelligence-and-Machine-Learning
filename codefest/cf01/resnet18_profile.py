"""
CF01 – ResNet18 Profiling + Roofline Plot
ECE 510, Spring 2026
Platform: CPU (PyTorch)

Usage:
    pip install torch torchvision fvcore matplotlib
    python resnet18_profile.py
"""

import cProfile
import pstats
import io
import time
import tracemalloc

import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 1
RUNS       = 20

# ── Load model ────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.eval()
x = torch.randn(BATCH_SIZE, 3, 224, 224)

# ── 1. Baseline benchmark ─────────────────────────────────────────────────────
def run_benchmark():
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(x)

    tracemalloc.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model(x)
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed  = t1 - t0
    per_pass = elapsed / RUNS
    samples_s = BATCH_SIZE / per_pass

    print("=" * 52)
    print("  ResNet18 Software Baseline Benchmark")
    print("=" * 52)
    print(f"  Platform        : CPU (torch, no CUDA)")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Runs            : {RUNS}")
    print(f"  Time / batch    : {per_pass*1e3:.2f} ms")
    print(f"  Throughput      : {samples_s:.2f} samples/sec")
    print(f"  Peak memory     : {peak_mem/1024:.1f} KB")
    print("=" * 52)
    return per_pass, samples_s, peak_mem

# ── 2. cProfile ───────────────────────────────────────────────────────────────
def run_profile():
    pr = cProfile.Profile()
    pr.enable()
    with torch.no_grad():
        for _ in range(RUNS):
            model(x)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(10)
    output = buf.getvalue()
    print("\n-- cProfile top-10 (cumulative) --")
    print(output)

    # Save to file
    with open("resnet18_profile.txt", "w") as f:
        f.write("ECE 510 – CF01 Profiling Output\n")
        f.write("Platform: CPU – PyTorch ResNet18, batch=1\n")
        f.write("=" * 52 + "\n")
        f.write(output)
    print("Saved: resnet18_profile.txt")
    return output

# ── 3. MAC count + Arithmetic Intensity ──────────────────────────────────────
def compute_arithmetic_intensity():
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_analysis = FlopCountAnalysis(model, x)
        total_macs = flops_analysis.total() / 2       # fvcore counts FLOPs = 2×MACs
        total_flops = total_macs * 2

        print(f"\n-- Arithmetic Intensity --")
        print(f"  Total MACs  : {total_macs/1e6:.1f} M")
        print(f"  Total FLOPs : {total_flops/1e9:.3f} GFLOP")

        # Memory traffic estimate (FP32, no caching)
        # Weights: 11.7M params × 4 bytes
        num_params = sum(p.numel() for p in model.parameters())
        bytes_weights = num_params * 4
        # Activations (rough): ~50 MB for ResNet18 at batch=1
        bytes_activations = 50 * 1024 * 1024
        total_bytes = bytes_weights + bytes_activations

        ai = total_flops / total_bytes
        print(f"  Weight mem  : {bytes_weights/1e6:.1f} MB")
        print(f"  Activation  : {bytes_activations/1e6:.1f} MB (estimate)")
        print(f"  Total bytes : {total_bytes/1e6:.1f} MB")
        print(f"  AI          : {ai:.2f} FLOPs/byte")
        return total_flops, ai

    except ImportError:
        print("fvcore not installed. Using manual estimate.")
        total_flops = 3.63e9   # known value for ResNet18 224x224
        total_bytes = 97e6
        ai = total_flops / total_bytes
        print(f"  Total FLOPs : {total_flops/1e9:.3f} GFLOP (manual)")
        print(f"  AI          : {ai:.2f} FLOPs/byte (manual)")
        return total_flops, ai

# ── 4. Roofline plot ──────────────────────────────────────────────────────────
def plot_roofline(ai, samples_s, total_flops):
    peak_compute   = 150.0   # GFLOP/s (Intel i7, AVX2, FP32 estimate)
    peak_bandwidth =  40.0   # GB/s (DDR4 dual-channel)
    ridge_point    = peak_compute / peak_bandwidth

    x_range  = np.logspace(-1, 3, 500)
    roofline = np.minimum(peak_bandwidth * x_range, peak_compute)

    # Achieved GFLOP/s from benchmark
    achieved = (total_flops / 1e9) * samples_s   # GFLOP/s at measured throughput

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x_range, roofline, 'b-', linewidth=2, label='Roofline (CPU)')
    ax.axvline(ridge_point, color='b', linestyle='--', alpha=0.4,
               label=f'Ridge point ({ridge_point:.1f} FLOPs/byte)')

    # Theoretical ceiling at this AI
    hw_ceiling = min(peak_bandwidth * ai, peak_compute)
    ax.plot(ai, hw_ceiling, 'r*', markersize=16,
            label=f'ResNet18 theoretical ceiling (AI={ai:.1f})')

    # Measured software performance
    ax.plot(ai, achieved, 'gs', markersize=12,
            label=f'Measured SW performance ({achieved:.2f} GFLOP/s)')
    ax.annotate('SW measured', xy=(ai, achieved),
                xytext=(ai * 0.3, achieved * 3),
                arrowprops=dict(arrowstyle='->', color='green'), color='green')

    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('Roofline Model – ResNet18 Inference on CPU', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0.1, 1000])
    ax.set_ylim([0.001, peak_compute * 2])

    plt.tight_layout()
    plt.savefig('roofline_resnet18.png', dpi=150)
    print("\nRoofline plot saved to roofline_resnet18.png")
    plt.show()

# ── 5. Top-5 MAC table (fvcore) ───────────────────────────────────────────────
def print_top5_macs():
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, x)
        by_module = fa.by_module()
        # Filter to conv layers only
        conv_layers = {k: v for k, v in by_module.items() if 'conv' in k.lower() and v > 0}
        sorted_layers = sorted(conv_layers.items(), key=lambda t: t[1], reverse=True)[:5]

        print("\n-- Top-5 layers by MACs --")
        print(f"{'Rank':<5} {'Layer':<30} {'MACs (M)':>10} {'% Total':>8}")
        print("-" * 58)
        total_flops = fa.total()
        for i, (name, flops) in enumerate(sorted_layers, 1):
            macs = flops / 2
            pct  = flops / total_flops * 100
            print(f"{i:<5} {name:<30} {macs/1e6:>10.1f} {pct:>7.1f}%")
    except ImportError:
        print("fvcore not installed – skipping MAC table.")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    per_pass, samples_s, peak_mem = run_benchmark()
    run_profile()
    total_flops, ai = compute_arithmetic_intensity()
    print_top5_macs()
    plot_roofline(ai, samples_s, total_flops)
