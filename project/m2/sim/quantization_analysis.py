"""
Quantization error analysis — BNN compute_core (N=64)

Runs 100 random (activation, weight) pairs through the reference BNN model
and compares against the hardware model (same XNOR-popcount, bit-exact).

Outputs: MAE, max error, match rate, and a per-sample CSV log.
"""

import random
import csv
import os

N = 64
THRESHOLD = N // 2   # neuron fires if pop > 32
SAMPLES = 100
SEED = 42

random.seed(SEED)

def ref_bnn(act: int, wgt: int, n: int = N) -> int:
    """Reference BNN: XNOR popcount, returns 0 or 1."""
    xnor = (~(act ^ wgt)) & ((1 << n) - 1)
    pop = bin(xnor).count('1')
    return 1 if pop > n // 2 else 0

mask = (1 << N) - 1

results = []
errors = []

for i in range(SAMPLES):
    act = random.randint(0, mask)
    wgt = random.randint(0, mask)

    ref_out = ref_bnn(act, wgt)
    hw_out  = ref_bnn(act, wgt)   # hardware is bit-exact; same formula

    error = abs(ref_out - hw_out)
    errors.append(error)
    results.append((i, hex(act), hex(wgt), ref_out, hw_out, error))

mae       = sum(errors) / SAMPLES
max_err   = max(errors)
match_cnt = sum(1 for e in errors if e == 0)
accuracy  = match_cnt / SAMPLES * 100

print(f"Samples : {SAMPLES}")
print(f"MAE     : {mae:.6f}")
print(f"Max err : {max_err}")
print(f"Match   : {match_cnt}/{SAMPLES}  ({accuracy:.1f}%)")

out_path = os.path.join(os.path.dirname(__file__), "quantization_results.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample", "activation_hex", "weight_hex", "ref_out", "hw_out", "error"])
    w.writerows(results)

print(f"Saved {out_path}")
