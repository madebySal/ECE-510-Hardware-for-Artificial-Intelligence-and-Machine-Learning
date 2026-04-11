"""
Pure-NumPy BNN inference for ECE 510 profiling.
Architecture: [784 -> 256 -> 128 -> 10] (same as CF01 MLP, but binary weights/activations).
Weights in {+1, -1}, activations binarized after each hidden layer.
XNOR-popcount path is simulated via packed uint64 bit arrays.
"""

import numpy as np
import time

SEED = 42
rng = np.random.default_rng(SEED)

def make_binary_weights(shape):
    return rng.choice([-1, 1], size=shape).astype(np.float32)

def pack_bits(x):
    """Pack a {+1,-1} vector into uint64 words (1=+1, 0=-1)."""
    bits = (x > 0).astype(np.uint8)
    pad = (64 - len(bits) % 64) % 64
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    words = np.packbits(bits, bitorder='little')
    # reinterpret as uint64
    return words.view(np.uint64)

def xnor_popcount(a_packed, w_packed):
    """
    XNOR + popcount between two packed bit vectors.
    Returns integer dot product in range [-N, +N].
    """
    xnor = ~(a_packed ^ w_packed)          # XNOR
    # popcount via bin(int).count('1') for each word
    ones = sum(bin(int(w)).count('1') for w in xnor)
    n_bits = len(a_packed) * 64
    return 2 * ones - n_bits               # convert popcount to {+1,-1} sum

def bnn_layer_float(x, W):
    """Standard float32 linear layer (baseline)."""
    return x @ W.T

def bnn_layer_xnor(x, W):
    """
    XNOR-popcount layer.
    x: (in_features,) float32 {+1,-1}
    W: (out_features, in_features) float32 {+1,-1}
    """
    out = np.empty(W.shape[0], dtype=np.float32)
    a_packed = pack_bits(x)
    for i in range(W.shape[0]):
        w_packed = pack_bits(W[i])
        out[i] = xnor_popcount(a_packed, w_packed)
    return out

def sign(x):
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)

# Network dims
DIMS = [784, 256, 128, 10]
W1 = make_binary_weights((DIMS[1], DIMS[0]))
W2 = make_binary_weights((DIMS[2], DIMS[1]))
W3 = make_binary_weights((DIMS[3], DIMS[2]))

def bnn_forward_float(x):
    h1 = bnn_layer_float(x, W1); a1 = sign(h1)
    h2 = bnn_layer_float(a1, W2); a2 = sign(h2)
    h3 = bnn_layer_float(a2, W3)
    return h3

def bnn_forward_xnor(x):
    a0 = sign(x)
    h1 = bnn_layer_xnor(a0, W1); a1 = sign(h1)
    h2 = bnn_layer_xnor(a1, W2); a2 = sign(h2)
    h3 = bnn_layer_xnor(a2, W3)
    return h3

if __name__ == "__main__":
    import cProfile, pstats, io

    x_sample = rng.standard_normal(784).astype(np.float32)

    # ---- cProfile on float path ----
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        bnn_forward_float(x_sample)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("=== cProfile: float path (10 runs) ===")
    print(s.getvalue())

    # ---- cProfile on xnor path ----
    pr2 = cProfile.Profile()
    pr2.enable()
    for _ in range(10):
        bnn_forward_xnor(x_sample)
    pr2.disable()

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2).sort_stats('cumulative')
    ps2.print_stats(20)
    print("=== cProfile: XNOR-popcount path (10 runs) ===")
    print(s2.getvalue())

    # ---- Wall-clock timing ----
    N = 10
    times_float, times_xnor = [], []
    for _ in range(N):
        t0 = time.perf_counter(); bnn_forward_float(x_sample); times_float.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); bnn_forward_xnor(x_sample);  times_xnor.append(time.perf_counter()-t0)

    print(f"Float path  — median: {np.median(times_float)*1e3:.3f} ms, min: {min(times_float)*1e3:.3f} ms")
    print(f"XNOR path   — median: {np.median(times_xnor)*1e3:.3f} ms, min: {min(times_xnor)*1e3:.3f} ms")

    # ---- Arithmetic intensity calculation ----
    # Layer 1: 784->256 binary XNOR-popcount
    N_in, N_out = 784, 256
    # FLOPs: 1 XNOR + 1 popcount add per input element, per output neuron
    # Each XNOR-popcount neuron: N_in XNORs + (N_in-1) adds = 2*N_in - 1 ops ≈ 2*N_in
    flops_layer1 = 2 * N_in * N_out
    # Bytes: weights (packed bits: 1 bit each -> N_in*N_out / 8 bytes),
    #        input (packed: N_in/8 bytes), output (N_out * 4 bytes FP32 pre-binarize)
    bytes_weights_l1 = (N_in * N_out) // 8
    bytes_input_l1   = N_in // 8
    bytes_output_l1  = N_out * 4   # float32 output before sign()
    bytes_total_l1   = bytes_weights_l1 + bytes_input_l1 + bytes_output_l1
    ai_l1 = flops_layer1 / bytes_total_l1

    print(f"\n=== Arithmetic Intensity: Layer 1 XNOR-popcount ({N_in}->{N_out}) ===")
    print(f"FLOPs = 2 × {N_in} × {N_out} = {flops_layer1:,}")
    print(f"Bytes weights = {N_in}×{N_out}/8 = {bytes_weights_l1:,} bytes")
    print(f"Bytes input   = {N_in}/8 = {bytes_input_l1} bytes")
    print(f"Bytes output  = {N_out}×4 = {bytes_output_l1:,} bytes")
    print(f"Total bytes   = {bytes_total_l1:,} bytes")
    print(f"AI = {flops_layer1:,} / {bytes_total_l1:,} = {ai_l1:.2f} FLOP/byte")

    # Full network
    layers = [(784,256),(256,128),(128,10)]
    total_flops, total_bytes = 0, 0
    for ni, no in layers:
        f = 2*ni*no
        b = (ni*no)//8 + ni//8 + no*4
        total_flops += f; total_bytes += b
    print(f"\n=== Full network AI ===")
    print(f"Total FLOPs  = {total_flops:,}")
    print(f"Total bytes  = {total_bytes:,}")
    print(f"Network AI   = {total_flops/total_bytes:.2f} FLOP/byte")
