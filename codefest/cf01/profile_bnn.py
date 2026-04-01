"""
CF01 – BNN Profiling + Roofline Plot
ECE 510, Spring 2026
"""

import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
from bnn_baseline import BNN, binarize, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

# ── 1. cProfile ───────────────────────────────────────────────────────────────
def profile_model():
    rng   = np.random.default_rng(0)
    x     = binarize(rng.standard_normal((BATCH_SIZE, INPUT_SIZE)).astype(np.float32))
    model = BNN()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        model.forward(x)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(10)
    print(buf.getvalue())

# ── 2. Arithmetic intensity ───────────────────────────────────────────────────
def compute_arithmetic_intensity():
    # FLOPs (treat XNOR+popcount as 2 ops: 1 mul + 1 add equivalent)
    flops_l1 = 2 * INPUT_SIZE  * HIDDEN_SIZE  # per sample
    flops_l2 = 2 * HIDDEN_SIZE * OUTPUT_SIZE

    # Bytes moved (binary packed: 1 bit per weight/activation)
    bytes_w1  = (INPUT_SIZE  * HIDDEN_SIZE)  / 8   # weights layer 1
    bytes_w2  = (HIDDEN_SIZE * OUTPUT_SIZE)  / 8   # weights layer 2
    bytes_in  =  INPUT_SIZE                  / 8   # input activations
    bytes_h   =  HIDDEN_SIZE                 / 8   # hidden activations
    bytes_out =  OUTPUT_SIZE * 4                   # output (float32)

    total_flops = flops_l1 + flops_l2
    total_bytes = bytes_w1 + bytes_w2 + bytes_in + bytes_h + bytes_out

    ai = total_flops / total_bytes
    print(f"Total FLOPs / sample : {total_flops:,}")
    print(f"Total bytes / sample : {total_bytes:.1f}")
    print(f"Arithmetic intensity : {ai:.2f} FLOPs/byte")
    return ai

# ── 3. Roofline plot ──────────────────────────────────────────────────────────
def plot_roofline(ai):
    # Laptop CPU (Intel Core i7 approx.)
    peak_compute   = 150.0   # GFLOP/s (with AVX2, FP32)
    peak_bandwidth =  40.0   # GB/s (DDR4 dual-channel)
    ridge_point    = peak_compute / peak_bandwidth  # FLOPs/byte

    x_range = np.logspace(-1, 3, 500)
    roofline = np.minimum(peak_bandwidth * x_range, peak_compute)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x_range, roofline, 'b-', linewidth=2, label='Roofline (CPU)')
    ax.axvline(ridge_point, color='b', linestyle='--', alpha=0.4, label=f'Ridge point ({ridge_point:.1f} FLOPs/byte)')

    # BNN kernel point
    # Measured ~0.004 GFLOP/s from baseline (software, not hardware)
    # We mark the theoretical hardware performance = min(roofline at AI)
    hw_perf = min(peak_bandwidth * ai, peak_compute)
    ax.plot(ai, hw_perf, 'r*', markersize=16, label=f'BNN kernel (AI={ai:.1f} FLOPs/byte)')
    ax.annotate('BNN kernel', xy=(ai, hw_perf),
                xytext=(ai * 2, hw_perf * 0.5),
                arrowprops=dict(arrowstyle='->', color='red'), color='red')

    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title('Roofline Model – BNN Inference on CPU', fontsize=13)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0.1, 1000])
    ax.set_ylim([0.1, peak_compute * 2])

    plt.tight_layout()
    plt.savefig('roofline_bnn.png', dpi=150)
    print("Roofline plot saved to roofline_bnn.png")
    plt.show()

# ── 4. Interface justification ────────────────────────────────────────────────
INTERFACE_JUSTIFICATION = """
Interface Selection: SPI
========================
The BNN inference chiplet targets edge deployment on an MCU-class host (e.g., ARM Cortex-M).
For a single-sample inference pass, the input data is 784 / 8 = 98 bytes and the weight
memory is (784x256 + 256x10) / 8 ~= 25,920 bytes -- small enough that SPI at ~50 Mbit/s
(~6.25 MB/s) can transfer a full weight load in ~4 ms and each input in <0.2 ms. Given
that BNN inference itself takes <1 ms in hardware, SPI does not become the bottleneck for
single-sample keyword-spotting or gesture-recognition workloads. The simplicity of SPI
also minimizes verification overhead and is directly supported by most MCU peripherals
without additional IP. AXI4 would offer higher bandwidth but adds significant interface
complexity unjustified at this data scale.
"""

if __name__ == "__main__":
    print("\n-- cProfile results --")
    profile_model()
    print("\n-- Arithmetic Intensity --")
    ai = compute_arithmetic_intensity()
    print("\n-- Interface Justification --")
    print(INTERFACE_JUSTIFICATION)
    print("\n-- Roofline Plot --")
    plot_roofline(ai)
