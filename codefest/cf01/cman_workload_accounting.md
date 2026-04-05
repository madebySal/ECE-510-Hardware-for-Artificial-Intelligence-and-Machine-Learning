Network: [784 → 256 → 128 → 10], batch=1, FP32 (4 bytes), no biases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(1) MACs per layer:
    Formula: MACs = input_size × output_size

    Layer 1: 784 × 256 = 200,704
    Layer 2: 256 × 128 =  32,768
    Layer 3: 128 × 10  =   1,280

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(2) Total MACs:
    200,704 + 32,768 + 1,280 = 234,752

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(3) Total Parameters (weights only, no biases):
    Layer 1: 784 × 256 = 200,704
    Layer 2: 256 × 128 =  32,768
    Layer 3: 128 × 10  =   1,280
                          ───────
    Total                = 234,752

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(4) Weight memory (FP32 = 4 bytes per parameter):
    234,752 × 4 = 939,008 bytes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(5) Activation memory (input + all layer outputs):
    Input:    784 × 4 =  3,136 bytes
    Layer 1:  256 × 4 =  1,024 bytes
    Layer 2:  128 × 4 =    512 bytes
    Layer 3:   10 × 4 =     40 bytes
                          ──────────
    Total              =  4,712 bytes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(6) Arithmetic Intensity:
    Formula: (2 × total MACs) / (weight bytes + activation bytes)

    = (2 × 234,752) / (939,008 + 4,712)
    = 469,504 / 943,720
    = 0.497 FLOPs/byte
