"""
CF01 – BNN Software Baseline Benchmark
ECE 510, Spring 2026
Platform: CPU (NumPy)
"""

import numpy as np
import time
import tracemalloc

# ── Model config ──────────────────────────────────────────────────────────────
INPUT_SIZE   = 784     # e.g. MNIST flattened
HIDDEN_SIZE  = 256
OUTPUT_SIZE  = 10
BATCH_SIZE   = 1000
RUNS         = 50      # number of timed inference passes

# ── Binary helpers ────────────────────────────────────────────────────────────
def binarize(x):
    """Binarize to +1 / -1."""
    return np.sign(x).astype(np.float32)

def binary_linear(x_bin, w_bin, bias=None):
    """
    Binary fully-connected layer.
    Implements XNOR-popcount as: out = (2 * popcount(XNOR(x, w))) - N
    Using float matmul as software stand-in; arithmetic intensity is computed
    separately based on the binary operation count.
    """
    out = x_bin @ w_bin.T          # shape: (batch, out)
    if bias is not None:
        out += bias
    return out

def batch_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean(axis=0)
    var = x.var(axis=0)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

# ── BNN model (2 binary layers) ───────────────────────────────────────────────
class BNN:
    def __init__(self):
        rng = np.random.default_rng(42)
        # Binary weights (+1 / -1)
        self.w1 = binarize(rng.standard_normal((HIDDEN_SIZE, INPUT_SIZE)).astype(np.float32))
        self.w2 = binarize(rng.standard_normal((OUTPUT_SIZE, HIDDEN_SIZE)).astype(np.float32))
        # BatchNorm params (real-valued)
        self.gamma1 = np.ones(HIDDEN_SIZE,  dtype=np.float32)
        self.beta1  = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.gamma2 = np.ones(OUTPUT_SIZE,  dtype=np.float32)
        self.beta2  = np.zeros(OUTPUT_SIZE, dtype=np.float32)

    def forward(self, x):
        # Layer 1
        h = binary_linear(x, self.w1)
        h = batch_norm(h, self.gamma1, self.beta1)
        h = binarize(h)
        # Layer 2
        out = binary_linear(h, self.w2)
        out = batch_norm(out, self.gamma2, self.beta2)
        return out  # raw logits

# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark():
    rng  = np.random.default_rng(0)
    x    = binarize(rng.standard_normal((BATCH_SIZE, INPUT_SIZE)).astype(np.float32))
    model = BNN()

    # Warm-up
    for _ in range(5):
        model.forward(x)

    # Timed runs
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        model.forward(x)
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed   = t1 - t0                          # seconds for RUNS passes
    per_pass  = elapsed / RUNS                   # seconds per batch
    samples_s = BATCH_SIZE / per_pass            # samples/sec

    # FLOPs per batch (2 × N × M per layer, ×2 for mul+add)
    flops_l1  = 2 * INPUT_SIZE  * HIDDEN_SIZE * BATCH_SIZE
    flops_l2  = 2 * HIDDEN_SIZE * OUTPUT_SIZE  * BATCH_SIZE
    total_flops = flops_l1 + flops_l2
    gflops_s  = (total_flops / per_pass) / 1e9

    print("=" * 50)
    print("  BNN Software Baseline Benchmark")
    print("=" * 50)
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Runs            : {RUNS}")
    print(f"  Time / batch    : {per_pass*1e3:.3f} ms")
    print(f"  Throughput      : {samples_s:,.0f} samples/sec")
    print(f"  Compute         : {gflops_s:.4f} GFLOP/s")
    print(f"  Peak memory     : {peak_mem/1024:.1f} KB")
    print("=" * 50)

    return per_pass, samples_s, gflops_s, peak_mem

if __name__ == "__main__":
    run_benchmark()
