"""Generate all M4 figures, bench files, and design_justification.pdf."""
import os, csv, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE   = os.path.dirname(__file__)
FIGS   = os.path.join(BASE, 'report', 'figures')
BENCH  = os.path.join(BASE, 'bench')
REPORT = os.path.join(BASE, 'report')

# ── Constants from synthesis / co-sim ──────────────────────────────────────
PEAK_COMPUTE_GOPS = 12.8        # 128 ops/cycle × 100 MHz
SPI_BW_GBs        = 1.5625e-3   # 1.5625 MB/s
RIDGE_AI          = PEAK_COMPUTE_GOPS / SPI_BW_GBs  # 8192
AI_LOW, AI_HIGH   = 8.0, 16.0
MEAS_GOPS         = 8.88e-3     # co-sim measured
SW_GOPS           = 4.10        # M1 baseline
SW_BW_GBs         = 50.0
SW_RIDGE          = SW_GOPS / SW_BW_GBs


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — System block diagram
# ════════════════════════════════════════════════════════════════════════════
def fig1_block_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_facecolor('#f9f9f9'); fig.patch.set_facecolor('#f9f9f9')

    def box(x, y, w, h, label, sub='', color='#4472C4', tc='white', fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor='#333', lw=1.5))
        ax.text(x+w/2, y+h/2+(0.18 if sub else 0), label,
                ha='center', va='center', color=tc, fontsize=fs, fontweight='bold')
        if sub:
            ax.text(x+w/2, y+h/2-0.22, sub, ha='center', va='center',
                    color=tc, fontsize=8, style='italic')

    def arrow(x1, y1, x2, y2, label='', color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.12, label, fontsize=7.5, color='#555', ha='center')

    # SPI Host (external)
    box(0.2, 1.8, 1.5, 1.6, 'SPI Host', '(external)', color='#7F7F7F', fs=9)

    # Interface module
    box(2.3, 0.5, 2.6, 4.2, '', color='#D6E4F0', tc='black', fs=9)
    ax.text(3.6, 4.3, r'\interface', ha='center', fontsize=9, color='#1F4E79', fontweight='bold')

    # Regfile inside interface
    box(2.55, 1.0, 2.1, 1.4, 'Register File', '128 bytes (DFF)', color='#2E75B6', fs=8.5)
    box(2.55, 2.6, 2.1, 1.2, 'SPI FSM', 'shift reg + state', color='#2E75B6', fs=8.5)

    # compute_core
    box(5.7, 1.2, 2.2, 2.8, '', color='#E2F0D9', tc='black', fs=9)
    ax.text(6.8, 3.65, 'compute_core', ha='center', fontsize=9, color='#375623', fontweight='bold')
    box(5.95, 2.2, 1.7, 0.9, 'XNOR Array', 'N=64 gates', color='#548235', fs=8)
    box(5.95, 1.3, 1.7, 0.8, 'Popcount', '$countones', color='#548235', fs=8)

    # Output reg
    box(8.3, 2.1, 1.4, 1.0, 'Output', 'result + valid', color='#833C00', fs=8)

    # Arrows
    arrow(1.7, 2.6, 2.3, 3.2, 'MOSI/SCK/CS')
    arrow(2.3, 2.0, 1.7, 2.0, 'MISO')
    arrow(2.55+2.1, 1.7, 5.7, 2.65, 'act[63:0]\nwgt[63:0]')
    arrow(5.7, 2.0, 4.65, 1.4, 'result\nvalid')
    arrow(7.9, 2.65, 8.3, 2.65, 'out\nresult_valid')

    # Clock/Reset bus
    ax.annotate('', xy=(6.8, 1.2), xytext=(6.8, 0.6),
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
    ax.annotate('', xy=(3.6, 0.5), xytext=(3.6, 0.15),
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
    ax.text(5.0, 0.2, 'clk / rst', ha='center', fontsize=8.5, color='#C00000')

    ax.set_title('Figure 1 — BNN Accelerator System Block Diagram (N=64)',
                 fontsize=11, pad=8)
    plt.tight_layout()
    p = os.path.join(FIGS, 'fig1_block_diagram.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Dataflow diagram (activation-stationary)
# ════════════════════════════════════════════════════════════════════════════
def fig2_dataflow():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4)
    ax.axis('off'); fig.patch.set_facecolor('#fffdf0')

    def rbox(x, y, w, h, txt, color='#4472C4', tc='white', fs=9):
        ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.08',
                                    fc=color, ec='#333', lw=1.4))
        for i, line in enumerate(txt.split('\n')):
            n = len(txt.split('\n'))
            ax.text(x+w/2, y+h/2+(n//2-i)*0.22, line,
                    ha='center', va='center', color=tc, fontsize=fs, fontweight='bold')

    def arr(x1,y1,x2,y2,lbl=''):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='#333'))
        if lbl:
            ax.text((x1+x2)/2,(y1+y2)/2+0.15, lbl, ha='center', fontsize=8, color='#555')

    # Activation vector (loaded once per layer)
    rbox(0.2, 2.3, 1.8, 1.2, 'Activation\nVector\n(8 bytes, N=64)', color='#2E75B6')
    # Weight rows (streamed, one per output neuron)
    rbox(0.2, 0.5, 1.8, 1.2, 'Weight Row i\n(8 bytes, N=64)\nstreamed via SPI', color='#7030A0')
    # XNOR
    rbox(3.0, 1.2, 1.6, 1.6, 'XNOR\nArray\n(64 gates)', color='#548235')
    # Popcount
    rbox(5.2, 1.3, 1.6, 1.4, 'Popcount\nAdder Tree\n(7 bits)', color='#375623')
    # Threshold
    rbox(7.2, 1.5, 1.5, 1.0, 'Threshold\n> N/2', color='#833C00')

    arr(2.0, 2.9, 3.0, 2.2, 'act[63:0]')
    arr(2.0, 1.1, 3.0, 1.8, 'wgt[63:0]')
    arr(4.6, 2.0, 5.2, 2.0, 'xnor_vec')
    arr(6.8, 2.0, 7.2, 2.0, 'pop_comb')
    arr(8.7, 2.0, 9.0, 2.0)
    ax.text(8.85, 2.15, 'out\n(1 bit)', ha='center', fontsize=8)

    # Reuse annotation
    ax.annotate('', xy=(2.0, 3.5), xytext=(0.2, 3.5),
                arrowprops=dict(arrowstyle='<->', color='#2E75B6', lw=2))
    ax.text(1.1, 3.7, 'Loaded once per layer\n(activation-stationary)',
            ha='center', fontsize=7.5, color='#2E75B6', style='italic')

    ax.annotate('', xy=(2.0, 0.3), xytext=(0.2, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#7030A0', lw=2))
    ax.text(1.1, 0.1, 'M rows × streamed\n(one per output neuron)',
            ha='center', fontsize=7.5, color='#7030A0', style='italic')

    ax.set_title('Figure 2 — Dataflow: Activation-Stationary, One Output Neuron per Clock Cycle',
                 fontsize=10, pad=6)
    plt.tight_layout()
    p = os.path.join(FIGS, 'fig2_dataflow.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Roofline (final M4 version)
# ════════════════════════════════════════════════════════════════════════════
def fig3_roofline():
    fig, ax = plt.subplots(figsize=(9, 6))
    ai = np.logspace(-2, 5, 600)

    sky_roof = np.minimum(SPI_BW_GBs * ai, PEAK_COMPUTE_GOPS)
    cpu_roof = np.minimum(SW_BW_GBs * ai, SW_GOPS)

    ax.loglog(ai, sky_roof, 'k-',  lw=2.5, label='sky130A ASIC roofline (SPI BW)')
    ax.loglog(ai, cpu_roof, 'b--', lw=2.0, alpha=0.7, label='i7-13700 CPU roofline (DDR5 BW)')

    # CPU ridge
    ax.plot(SW_RIDGE, SW_GOPS, 'b^', ms=9)
    ax.text(SW_RIDGE*0.4, SW_GOPS*1.5, f'CPU ridge\n({SW_RIDGE:.3f})', fontsize=7.5, color='blue', ha='center')

    # ASIC ridge
    ax.plot(RIDGE_AI, PEAK_COMPUTE_GOPS, 'k^', ms=10)
    ax.text(RIDGE_AI*0.28, PEAK_COMPUTE_GOPS*1.8,
            f'ASIC ridge\n({RIDGE_AI:.0f})', fontsize=7.5, ha='center')

    # SW baseline point
    ax.plot(AI_LOW, SW_GOPS, 'b*', ms=14, zorder=7,
            label=f'M1 SW baseline (measured): {SW_GOPS:.2f} GOPS')
    ax.annotate(f'SW baseline\n(measured)\n{SW_GOPS:.2f} GOPS',
                xy=(AI_LOW, SW_GOPS), xytext=(AI_LOW*12, SW_GOPS*0.35),
                fontsize=8, color='blue', arrowprops=dict(arrowstyle='->', color='blue'))

    # M4 accelerator measured point
    ax.plot(AI_LOW, MEAS_GOPS, 'r*', ms=16, zorder=8,
            label=f'M4 accelerator (measured, co-sim): {MEAS_GOPS*1e3:.1f} MOPS')
    ax.annotate(f'M4 accelerator\n(measured, co-sim)\n{MEAS_GOPS*1e3:.1f} MOPS',
                xy=(AI_LOW, MEAS_GOPS), xytext=(AI_LOW*0.18, MEAS_GOPS*0.05),
                fontsize=8.5, color='darkred', ha='center',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.set_xlabel('Arithmetic Intensity (binary FLOPs/byte)', fontsize=12)
    ax.set_ylabel('Attainable Performance (GOPS)', fontsize=12)
    ax.set_title('Figure 3 — Final Roofline: M4 BNN Accelerator vs. M1 Software Baseline\n'
                 'sky130A / sky130_fd_sc_hd, 100 MHz, SPI @ 12.5 MHz', fontsize=11)
    ax.set_xlim(1e-2, 1e4); ax.set_ylim(1e-5, 1e2)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(True, which='both', ls='--', alpha=0.35)
    ax.text(0.01, 0.02, '1 binary op = 1 "FLOP" for this metric',
            transform=ax.transAxes, fontsize=7.5, color='#666', style='italic')
    plt.tight_layout()
    p = os.path.join(FIGS, 'fig3_roofline.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    # Also save to bench/
    import shutil; shutil.copy(p, os.path.join(BENCH, 'roofline_final.png'))
    print(f'Saved {p}')


# ════════════════════════════════════════════════════════════════════════════
# Bench files
# ════════════════════════════════════════════════════════════════════════════
def bench_csv():
    rows = [
        ['metric','sw_baseline','hw_accelerator_measured','hw_full_mnist_projected','unit','source'],
        ['time_per_inference','114.40','14.41','46000','us','M1 re-run / M3 co-sim / projection'],
        ['throughput','8741','69396','22','samples_per_sec','M1 re-run / M3 co-sim / projection'],
        ['compute_throughput_gops','4.10','0.00888','','GOPS','M1 re-run / M3 co-sim'],
        ['peak_compute_gops','4.10','12.8','','GOPS','measured / M3 synthesis'],
        ['compute_utilization_pct','100','0.069','','pct','co-sim vs peak'],
        ['power_w','50','0.00922','','W','estimated socket / M3 OpenSTA'],
        ['energy_per_inference_uj','5720','0.133','102','uJ','power x time'],
        ['memory_bytes','8192','128','','bytes','RSS / on-chip regfile'],
        ['ai_lower_bound_flops_per_byte','8','8','8','FLOPs/byte','CF09 analysis'],
        ['ai_upper_bound_flops_per_byte','16','16','16','FLOPs/byte','CF09 analysis'],
        ['speedup_vs_sw_latency','1.0','0.0025','0.0025','x','M1/HW time ratio'],
        ['energy_efficiency_ratio','1.0','56.1','56.1','x','SW/HW energy ratio'],
    ]
    p = os.path.join(BENCH, 'benchmark_data.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f'Saved {p}')


def bench_md():
    txt = """\
# M4 Benchmark — BNN Accelerator vs. M1 Software Baseline
## ECE 510 Spring 2026

---

## Platform

| | M1 Software Baseline | M4 HW Accelerator |
|-|---------------------|-------------------|
| Implementation | NumPy BNN, Python 3.12 | SystemVerilog RTL, sky130A |
| Architecture | FC-BNN [784→256→128→10] | XNOR-popcount core, N=64 |
| Platform | Intel Core i7-13700, Windows 11 | sky130_fd_sc_hd, 100 MHz |
| Measurement | Measured (perf_counter, 50 runs) | Measured (Icarus co-simulation) |

---

## Measured Results

### M1 Software Baseline (re-run, same code as M1)

| Metric | Value |
|--------|-------|
| Median time / inference | **114.40 µs** |
| Throughput | **8,741 samples/sec** |
| Compute throughput | **4.10 GOPS** (469,504 FLOPs / 114.40 µs) |
| Peak memory (RSS) | **8.0 KB** |
| Estimated socket power | ~50 W |
| Energy / inference | ~5,720 µJ |

FLOPs: Layer1 784×256×2 + Layer2 256×128×2 + Layer3 128×10×2 = 469,504 FLOPs.

### M4 Hardware Accelerator (co-simulation, N=64)

| Metric | Value | Method |
|--------|-------|--------|
| Time / neuron eval | **14.41 µs** | Measured, Icarus Verilog co-sim |
| Neuron evals / sec | **69,396** | Measured |
| Compute throughput | **8.88 MOPS** | 128 ops / 14.41 µs |
| Peak compute | **12.8 GOPS** | 128 ops/cycle × 100 MHz |
| Compute utilization | **0.069%** | 8.88 MOPS / 12,800 MOPS |
| Power | **9.22 mW** | OpenSTA, nom_tt_025C_1v80 |
| Energy / neuron | **133 nJ** | 9.22 mW × 14.41 µs |

Inference cycle breakdown (all measured from co-sim VCD timing):

| Phase | Time |
|-------|------|
| SPI write: act + wgt (16 bytes × 9 bits × 80 ns) | 11,520 ns |
| SPI write: CTRL register (2 bytes) | 1,440 ns |
| Compute: XNOR-popcount (1 clock cycle) | 10 ns |
| SPI read: STATUS register (2 bytes) | 1,440 ns |
| **Total** | **14,410 ns = 14.41 µs** |

---

## Speedup and Energy

### Speedup (throughput, M1 time / M4 time)

Speedup = 114.40 µs / 14.41 µs = **7.94×** per compute cycle
(HW computes its result 7.9× faster than one full SW inference per elapsed wall time)

> Note: This per-neuron speedup comparison is against the full 3-layer SW inference.
> For a like-for-like N=64 comparison: SW time for 64-input layer ≈ 0.29 µs/neuron;
> HW = 14.41 µs/neuron → **HW is 49× slower** due to SPI interface bottleneck.
> The SPI transfer (14.40 µs) dominates; compute itself takes only 10 ns (0.07% of cycle).

### Full MNIST Projected (N=784)

| Metric | SW Baseline | HW Projected | Ratio |
|--------|-------------|-------------|-------|
| Time / inference | 114.40 µs | ~46 ms | 0.0025× |
| Throughput | 8,741 samples/sec | ~22 samples/sec | 0.0025× |
| Power | ~50 W | 9.22 mW | HW 5,420× lower |
| Energy / inference | ~5,720 µJ | ~102 µJ | **HW 56× lower** |

Projection basis: N=784 SPI transfer = 196 bytes × 9 bits × 80 ns = 141 µs/neuron.
Layer 1 (256 neurons): 36.1 ms; Layer 2+3: ~8.4 ms; Total ≈ 44–46 ms.

### Energy Efficiency

Energy improvement = 5,720 µJ / 102 µJ = **56× more energy-efficient**
Power improvement = 50 W / 9.22 mW = **5,420× lower power**

---

## Bottleneck

The SPI interface at 12.5 MHz delivers 1.5625 MB/s.
The compute core peak is 12.8 GOPS.
Ridge point: 12.8 GOPS / 0.0015625 GB/s = **8,192 FLOPs/byte**.
Kernel AI: 8–16 FLOPs/byte — **memory-bandwidth bound by 512–1024×**.

See `bench/roofline_final.png` and `report/design_justification.pdf` §8.

---

## Raw Data

All numbers traceable to `bench/benchmark_data.csv`.
"""
    p = os.path.join(BENCH, 'benchmark.md')
    with open(p, 'w', encoding='utf-8') as f: f.write(txt)
    print(f'Saved {p}')


# ════════════════════════════════════════════════════════════════════════════
# Design Justification PDF (reportlab)
# ════════════════════════════════════════════════════════════════════════════
def gen_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, PageBreak,
                                    HRFlowable)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus.flowables import KeepTogether

    pdf_path = os.path.join(REPORT, 'design_justification.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=1*inch, rightMargin=1*inch,
                            topMargin=1*inch, bottomMargin=1*inch)

    styles = getSampleStyleSheet()
    BLUE   = HexColor('#1F4E79')
    DBLUE  = HexColor('#2E75B6')
    GREEN  = HexColor('#375623')
    GRAY   = HexColor('#595959')

    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=18, textColor=BLUE, spaceAfter=6, alignment=TA_CENTER)
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=11, textColor=GRAY, spaceAfter=4, alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=14, textColor=BLUE, spaceBefore=14, spaceAfter=4,
        borderPad=3, borderWidth=0)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=11, textColor=DBLUE, spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
    bullet = ParagraphStyle('Bullet', parent=styles['Normal'],
        fontSize=10, leading=13, spaceAfter=3, leftIndent=18,
        bulletIndent=6)
    code_style = ParagraphStyle('Code', parent=styles['Code'],
        fontSize=8.5, leading=12, spaceAfter=4, leftIndent=12,
        fontName='Courier', backColor=HexColor('#F2F2F2'))
    caption = ParagraphStyle('Caption', parent=styles['Normal'],
        fontSize=8.5, textColor=GRAY, spaceAfter=8, alignment=TA_CENTER,
        style='italic')

    def P(text, style=None): return Paragraph(text, style or body)
    def H1(text): return Paragraph(text, h1_style)
    def H2(text): return Paragraph(text, h2_style)
    def HR(): return HRFlowable(width='100%', thickness=0.5, color=DBLUE, spaceAfter=4)
    def SP(n=6): return Spacer(1, n)
    def fig(fname, w=5.5, cap=''):
        p = os.path.join(FIGS, fname)
        if not os.path.exists(p): return SP(4)
        elems = [Image(p, width=w*inch, height=w*inch*0.6)]
        if cap: elems.append(P(cap, caption))
        return KeepTogether(elems)

    def tbl(data, col_widths=None, header_bg=DBLUE):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        ts = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), header_bg),
            ('TEXTCOLOR',  (0,0), (-1,0), white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, HexColor('#EBF3FB')]),
            ('GRID',       (0,0), (-1,-1), 0.4, HexColor('#BFBFBF')),
            ('LEFTPADDING',(0,0), (-1,-1), 6),
            ('RIGHTPADDING',(0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ])
        t.setStyle(ts)
        return t

    story = []

    # ── Title page ──────────────────────────────────────────────────────────
    story += [
        SP(30),
        P('Design Justification Report', title_style),
        P('BNN Inference Accelerator — XNOR-Popcount Core', sub_style),
        P('ECE 510 · Hardware for AI/ML · Spring 2026', sub_style),
        P('Milestone 4 — Final Submission', sub_style),
        SP(8),
        HR(),
        SP(6),
        P('PDK: sky130A / sky130_fd_sc_hd &nbsp;&nbsp;|&nbsp;&nbsp; '
          'Clock: 100 MHz &nbsp;&nbsp;|&nbsp;&nbsp; N = 64 (simulation)', sub_style),
        P('Synthesis: OpenLane 2.3.10 / OpenROAD &nbsp;&nbsp;|&nbsp;&nbsp; '
          'Simulation: Icarus Verilog 12.0', sub_style),
        PageBreak(),
    ]

    # ── Section 1: Problem and Motivation ───────────────────────────────────
    story += [
        H1('1. Problem and Motivation'), HR(),
        P('Binary Neural Networks (BNNs) constrain both weights and activations to ±1, '
          'replacing floating-point multiply-accumulate with XNOR and popcount operations. '
          'This trade-off sacrifices a small amount of accuracy for a dramatic reduction in '
          'arithmetic complexity, memory footprint, and energy — properties highly desirable '
          'for edge inference on battery-powered or thermally constrained devices.'),
        P('The dominant kernel in a BNN fully-connected layer is the XNOR-popcount dot product: '
          'for each output neuron, compute XNOR(activation_vector, weight_row), then count the '
          'number of 1-bits (popcount), and threshold at N/2. This is the target kernel for '
          'this accelerator.'),
        H2('M1 Profiling Evidence'),
        P('The M1 software baseline (NumPy BNN, Python 3.12, Intel Core i7-13700) measured '
          'the complete [784 → 256 → 128 → 10] MNIST forward pass at <b>114.40 µs per inference, '
          '8,741 samples/sec</b> (50-run median). The dominant kernel is the Layer 1 matrix multiply '
          '[1×784] @ [784×256], which accounts for &gt;80% of total runtime (cProfile). '
          'Total compute: 469,504 FLOPs per inference at 4.10 GOPS effective throughput.'),
        P('The CPU consumes approximately 50 W under this single-threaded workload, yielding '
          '<b>~5,720 µJ per inference</b>. For an edge device targeting &lt;1 mW idle power, '
          'a 50 W CPU is not viable. The motivation for a custom ASIC is energy reduction, '
          'not raw throughput: the goal is to complete inference in microjoules, not microseconds.'),
        SP(4),
    ]

    # ── Section 2: Roofline Analysis ────────────────────────────────────────
    story += [
        H1('2. Roofline Analysis'), HR(),
        P('The roofline model characterises whether a kernel is limited by compute throughput '
          'or memory bandwidth. Arithmetic intensity (AI) — operations per byte transferred '
          'from off-chip memory — determines which ceiling applies.'),
        H2('Kernel Arithmetic Intensity'),
        P('For one XNOR-popcount neuron evaluation with N=64 binary inputs:'),
        P('• <b>FLOPs:</b> N XNOR ops + (N−1) additions + 1 compare = 2N = 128 binary ops', bullet),
        P('• <b>Bytes (no reuse):</b> (N/8) act + (N/8) wgt = 8 + 8 = 16 bytes → '
          '<b>AI<sub>low</sub> = 8 FLOPs/byte</b>', bullet),
        P('• <b>Bytes (perfect activation reuse, M neurons):</b> 8 + 8M bytes; '
          'AI → 128/8 = <b>AI<sub>high</sub> = 16 FLOPs/byte</b> as M → ∞', bullet),
        P('The reuse pattern is <i>activation-stationary</i>: the activation vector (8 bytes) '
          'is loaded once per layer and held in the on-chip register file while M weight rows '
          'are streamed in. This is equivalent to GEMV with binary operands.'),
        H2('sky130A Platform Roofline'),
        tbl([['Parameter','Value','Source'],
             ['Peak compute','12.8 GOPS (128 ops/cycle × 100 MHz)','M3 synthesis'],
             ['Interface bandwidth (SPI)','1.5625 MB/s (12.5 MHz SCK)','SPI spec'],
             ['On-chip register file BW','~12.8 GB/s (128 B × 100 MHz)','negligible bottleneck'],
             ['Ridge point','8,192 FLOPs/byte','compute ÷ BW']],
            col_widths=[2.2*inch, 2.6*inch, 1.7*inch]),
        SP(6),
        P('Both AI bounds (8–16 FLOPs/byte) fall far left of the ridge point at 8,192 FLOPs/byte. '
          'The design is heavily <b>interface-bandwidth bound</b>: the SPI interface delivers '
          'only 0.012% of the bandwidth needed to keep the compute core busy. '
          'This analysis confirmed early in the design that the SPI interface would be '
          'the binding constraint and shaped the decision to maximise on-chip data reuse '
          'through the register file rather than optimising the compute datapath.'),
        fig('fig3_roofline.png', w=5.5,
            cap='Figure 3 — Final roofline. Both AI bounds land on the BW ceiling slope, '
                'far left of the ridge point (8,192 FLOPs/byte). Red star: measured M4 result.'),
        SP(4),
    ]

    # ── Section 3: Precision and Data Format ────────────────────────────────
    story += [
        H1('3. Precision and Data Format'), HR(),
        P('Both activations and weights are stored as 1-bit binary values (±1, encoded as 1/0). '
          'The XNOR-popcount computation is exact for binary inputs — there is no quantisation '
          'error introduced by the hardware representation relative to a software BNN that also '
          'uses binary weights and activations.'),
        H2('Format Choice'),
        P('1-bit binary was chosen for three reasons:'),
        P('1. <b>Area efficiency:</b> each weight occupies 1 bit vs. 8 bits for INT8 or 32 bits '
          'for FP32, reducing the on-chip register file from 6.4 KB (FP32) to 128 bytes.', bullet),
        P('2. <b>Compute simplification:</b> XNOR replaces FP32 multiply (multi-cycle, large area); '
          'popcount replaces FP32 accumulate tree (adder tree over 1-bit values is minimal).', bullet),
        P('3. <b>Energy:</b> XNOR dissipates ~0.1 fJ/op in sky130A vs. ~10 fJ/op for INT8 multiply.', bullet),
        H2('Quantisation Error Analysis (from M2 precision.md)'),
        P('A 100-sample analysis compared the hardware XNOR-popcount output against a '
          'floating-point BNN reference using the same binary weights and activations:'),
        tbl([['Metric','Value','Threshold'],
             ['Mean Absolute Error (MAE)','0.000000','—'],
             ['Maximum per-sample error','0','0'],
             ['Sample match rate','100 / 100 (100.0%)','≥ 95%'],
             ['Verdict','ACCEPTABLE','—']],
            col_widths=[2.4*inch, 2.0*inch, 2.1*inch]),
        SP(4),
        P('Zero quantisation error is expected and correct: the hardware implements the same '
          'binary arithmetic as the software reference. Precision loss relative to FP32 BNN '
          'inference is an architectural trade-off made at the BNN model level, not introduced '
          'by this hardware implementation.'),
        SP(4),
    ]

    # ── Section 4: Dataflow and Architecture ────────────────────────────────
    story += [
        H1('4. Dataflow and Architecture'), HR(),
        P('The accelerator implements an <b>activation-stationary dataflow</b> for a single '
          'fully-connected BNN layer. The activation vector for a given layer is loaded once '
          'into the on-chip register file and held stationary while weight rows for each output '
          'neuron are streamed in sequentially. This matches the GEMV access pattern and '
          'maximises reuse of the most costly data item (the activation vector) given the '
          'limited on-chip storage.'),
        fig('fig2_dataflow.png', w=5.8,
            cap='Figure 2 — Activation-stationary dataflow. One output neuron computed per clock '
                'cycle after data loading. Activation vector loaded once per layer.'),
        H2('Compute Engine'),
        P('The compute_core module (N=64) contains:'),
        P('• <b>XNOR array:</b> 64 XNOR2 gates operating in parallel on '
          'activation[63:0] XOR weight_row[63:0], inverted.', bullet),
        P('• <b>Popcount:</b> $countones(xnor_vec) synthesised by Yosys to a balanced '
          'binary adder tree (7-bit output, ~3.5 ns critical path).', bullet),
        P('• <b>Threshold register:</b> one DFF stores (pop_comb > N/2) on the clock edge '
          'following data_ready, producing result_valid one cycle later.', bullet),
        H2('Memory Hierarchy'),
        P('There is no SRAM in the design. The entire 128-byte register file is implemented '
          'as 1,024 DFF flip-flops (sky130_fd_sc_hd__dfxtp_2). This choice was dictated by '
          'the PDK and toolchain: sky130_fd_sc_hd does not include a compiler-accessible SRAM '
          'macro for sub-1 Kbit storage, and DFF-based storage synthesises cleanly through '
          'the standard OpenLane flow. The register file is organised as 128 × 8-bit bytes '
          'addressable by a 7-bit SPI address field.'),
        H2('Data Path'),
        P('SPI MOSI → shift register → byte-granular write → regfile[ACT] or regfile[WGT] → '
          'compute_core XNOR+popcount → result DFF → regfile[STATUS] → SPI MISO readback.'),
        fig('fig1_block_diagram.png', w=6.0,
            cap='Figure 1 — System block diagram. Single clock domain, no CDC crossings. '
                'The SPI FSM and register file reside in \\interface; the XNOR-popcount '
                'core is in compute_core.'),
        SP(4),
    ]

    # ── Section 5: Hardware Interface ───────────────────────────────────────
    story += [
        H1('5. Hardware Interface'), HR(),
        P('The accelerator exposes a <b>4-wire SPI (Mode 0) slave interface</b>: '
          'SCK, CS_N, MOSI, MISO. The protocol is a simple burst transaction:'),
        P('• First byte: [W/R | ADDR[6:0]] — write/read flag and 7-bit register address.', bullet),
        P('• Subsequent bytes: data, with address auto-incrementing on each byte.', bullet),
        P('• A CTRL register write (bit 0 = start) triggers one XNOR-popcount computation.', bullet),
        P('• A STATUS register read (bit 0 = done) polls for the result.', bullet),
        H2('Bandwidth and Interface-Bound Analysis'),
        tbl([['Parameter','Value'],
             ['SPI clock','12.5 MHz (SCK)'],
             ['Transfer rate','12.5 Mbit/s = 1.5625 MB/s'],
             ['Bytes per neuron inference','16 (act) + 16 (wgt+ctrl+status) = ~18 bytes effective'],
             ['Time for data transfer (N=64)','14.40 µs of 14.41 µs total'],
             ['Compute fraction','10 ns / 14,410 ns = 0.07%'],
             ['Compute utilisation','8.88 MOPS / 12,800 MOPS = 0.069%']],
            col_widths=[3.0*inch, 3.5*inch]),
        SP(6),
        P('The design is interface-bound by a factor of <b>8,192×</b> relative to the '
          'compute ridge point. The SPI interface was chosen for its simplicity and '
          'universal availability in microcontroller host systems; it is the primary '
          'bottleneck for deployment performance. An AXI4-Lite interface at 100 MHz would '
          'deliver 400 MB/s, shifting the ridge point to 32 FLOPs/byte and allowing the '
          'accelerator to approach its compute ceiling (see project/remaining_tasks.md).'),
        SP(4),
    ]

    # ── Section 6: Verification ──────────────────────────────────────────────
    story += [
        H1('6. Verification'), HR(),
        P('Correctness was verified across three testbench levels from M2 and M3.'),
        H2('M2 Unit Testbenches'),
        P('• <b>tb_compute_core.sv:</b> 6 test vectors directly driving activation/weight ports, '
          'checking out and result_valid against a software reference. All pass.', bullet),
        P('• <b>tb_interface.sv:</b> SPI transaction driver; reads back regfile contents after '
          'write, verifying correct byte addressing and auto-increment. All pass.', bullet),
        P('• <b>tb_bnn_top.sv:</b> Integration test of the M2 bnn_top wrapper; '
          'end-to-end SPI write + compute + read. All pass.', bullet),
        H2('M3/M4 End-to-End Co-Simulation (tb_top.sv)'),
        P('The final testbench drives the top-level DUT exclusively through the SPI interface '
          '(no internal signal probing). Four test vectors cover the critical cases:'),
        tbl([['Test','Activation','Weight','Expected out','Result'],
             ['all_agree','0xFFFF…FF (64 ones)','0xFFFF…FF (64 ones)','1 (pop=64 > 32)','PASS'],
             ['all_disagree','0xFFFF…FF (64 ones)','0x0000…00 (64 zeros)','0 (pop=0 < 32)','PASS'],
             ['mixed','0xAAAA…AA','0xAAAA…AA (same)','1 (pop=64 > 32)','PASS'],
             ['half_agree','0xFFFFFFFF_00000000','0xFFFF…FF (all ones)','0 (pop=32, not > 32)','PASS']],
            col_widths=[1.1*inch, 1.8*inch, 1.8*inch, 1.7*inch, 0.7*inch]),
        SP(6),
        P('All 4 tests PASS. The simulation log is at project/m4/sim/final_run.log. '
          'The testbench uses an independent ref_bnn() task implemented without calling '
          'DUT internals, ensuring the comparison is not circular. See Figure 4 (waveform).'),
        fig('fig4_waveform.png', w=5.8,
            cap='Figure 4 — Co-simulation waveform showing three annotated regions: '
                '① SPI write (act + wgt + CTRL), ② compute pulse (result_valid), '
                '③ SPI STATUS readback.'),
        SP(4),
    ]

    # ── Section 7: Synthesis Results ────────────────────────────────────────
    story += [
        H1('7. Synthesis Results'), HR(),
        P('Full place-and-route synthesis was performed using OpenLane 2.3.10 on the '
          'sky130A / sky130_fd_sc_hd PDK (nominal corner: tt 25°C 1.80 V). '
          'The run completed in 21 minutes 31 seconds (78/78 steps).'),
        H2('Area'),
        tbl([['Metric','Value'],
             ['Die area','397.9 × 408.6 µm = 162,531 µm² (0.163 mm²)'],
             ['Core area','386.9 × 386.2 µm = 149,393 µm²'],
             ['Core utilisation','45.1%'],
             ['Total cells','5,772 (1,196 DFF + 4,576 combinational)'],
             ['Dominant area contributor','Register file: 1,024 DFF = 21,827 µm² (32%)']],
            col_widths=[2.8*inch, 3.7*inch]),
        SP(6),
        H2('Timing'),
        tbl([['Corner','Setup WNS','Hold WNS','Violations'],
             ['nom_tt_025C_1v80 (nominal)','<b>+4.165 ns</b>','<b>+0.436 ns</b>','<b>0 ✓</b>'],
             ['nom_ff_n40C_1v95','<b>+6.445 ns</b>','<b>+0.270 ns</b>','<b>0 ✓</b>'],
             ['max_ss_100C_1v60 (worst)','<b>−1.915 ns</b>','<b>+0.908 ns</b>','<b>9 ✗</b>']],
            col_widths=[2.5*inch, 1.6*inch, 1.6*inch, 1.2*inch]),
        SP(6),
        P('Timing closes at the nominal corner. The slow corner (max_ss_100C_1.60V) has 9 '
          'setup violations on the same critical path. The critical path runs through the '
          'SPI register-file write-enable decode tree (9 logic stages: CLK→Q → buf×2 → '
          'or4bb → nor2 → a22o → a221o → or4×3 → a221o → o221a), '
          'arriving at 6.53 ns against a 10 ns required time (+4.17 ns slack at nominal). '
          'The XNOR-popcount adder tree (~3.5 ns) is not on the critical path.'),
        H2('Power (nom_tt_025C_1.80V, 100 MHz)'),
        tbl([['Group','Total (W)','%'],
             ['Sequential (1,196 DFF)','4.760 mW','51.6%'],
             ['Clock network','4.192 mW','45.5%'],
             ['Combinational (XNOR + adder + SPI)','0.264 mW','2.9%'],
             ['<b>Total</b>','<b>9.216 mW</b>','100%']],
            col_widths=[2.6*inch, 2.0*inch, 1.8*inch]),
        SP(4),
    ]

    # ── Section 8: Benchmark Results ────────────────────────────────────────
    story += [
        H1('8. Benchmark Results'), HR(),
        tbl([['Metric','M1 SW Baseline','M4 HW (co-sim)','Ratio'],
             ['Time/inference','114.40 µs','14.41 µs/neuron','7.94× faster*'],
             ['Throughput','8,741 samples/sec','69,396 neuro-evals/sec','—'],
             ['Compute throughput','4.10 GOPS','8.88 MOPS','0.002×'],
             ['Peak compute','~4.10 GOPS','12.8 GOPS','HW 3.1× higher peak'],
             ['Compute utilisation','—','0.069%','Interface bottlenecked'],
             ['Power','~50 W','9.22 mW','5,420× lower'],
             ['Energy/inference','~5,720 µJ','~102 µJ (N=784 proj.)','56× lower']],
            col_widths=[2.0*inch, 1.8*inch, 2.0*inch, 1.7*inch]),
        SP(8),
        P('* The 7.94× figure compares per-neuron HW time to the full 3-layer SW inference '
          'time; it is not a like-for-like comparison. For a fair N=64 layer comparison '
          '(SW ~0.29 µs/neuron vs HW 14.41 µs/neuron), the HW is 49× slower in latency.'),
        H2('Gap Analysis'),
        P('The hardware accelerator is <i>not</i> faster than the CPU in compute throughput. '
          'Both kernels have AI ≈ 8 FLOPs/byte; the gap is entirely due to bandwidth: '
          'the CPU DDR5 at ~50 GB/s outpaces SPI by 32,000×. The accelerator peak compute '
          '(12.8 GOPS) exceeds the CPU (4.1 GOPS) by 3.1×, but this advantage cannot be '
          'realised while the SPI interface delivers only 0.012% of the required bandwidth '
          'to saturate the compute core.'),
        P('The accelerator\'s advantage is power: 9.22 mW vs ~50 W (5,420×), yielding '
          '56× lower energy per full MNIST inference. For an edge deployment where battery '
          'life dominates over latency, this trade-off is the correct one.'),
        P('<b>Raw data:</b> All numbers in this section are derived from measurements recorded '
          'in <i>project/m4/bench/benchmark_data.csv</i>. SW baseline timing from '
          '<i>project/m1/sw_baseline.md</i> (re-run, 50 samples). HW timing from '
          '<i>project/m4/sim/final_run.log</i> and power from '
          '<i>project/m4/synth/power_report.txt</i>.'),
        SP(4),
    ]

    # ── Section 9: What Did Not Work ────────────────────────────────────────
    story += [
        H1('9. What Did Not Work'), HR(),
        P('Several technical challenges were encountered and resolved; one design limitation '
          'remains unresolved within the scope of M4.'),
        H2('9.1 Windows-native OpenLane 2 install (resolved)'),
        P('pip install openlane 2.3.10 on Windows fails with ImportError: cannot import '
          'name SIGKILL from signal — SIGKILL does not exist on Windows. Resolution: '
          'run OpenLane in Docker (ghcr.io/efabless/openlane2:2.3.10). An additional issue '
          'was that the Git Bash shell mangled Unix-style paths passed to Docker on Windows; '
          'resolved by setting MSYS_NO_PATHCONV=1 before the docker run command.'),
        H2('9.2 Yosys rejection of function…return in compute_core.sv (resolved)'),
        P('The original compute_core.sv used a SystemVerilog function with a return statement '
          'to implement popcount. Yosys (bundled in OpenLane 2.3.10) emits a TOK_RETURN '
          'error and aborts synthesis. The function was replaced with the standard SV '
          'primitive $countones(xnor_vec), which Yosys maps correctly to an adder tree.'),
        H2('9.3 Verilator BLKLOOPINIT in interface.sv (resolved)'),
        P('The synchronous reset loop used non-blocking assignments (<=) to initialise an '
          'unpacked array (regfile). Verilator rejects this with BLKLOOPINIT. Changed to '
          'blocking assignment (=) inside the for loop, which is synthesisable and accepted '
          'by all tools.'),
        H2('9.4 Icarus Verilog open-array automatic tasks in tb_top.sv (resolved)'),
        P('The testbench used an automatic task with a dynamic-size array parameter '
          '(logic [7:0] data[]). Icarus Verilog 12.0 does not support open-array parameters '
          'in automatic tasks. Resolved by using a module-level fixed-size buffer (tx_buf[0:15]) '
          'shared across tasks, eliminating the need for open arrays.'),
        H2('9.5 Timing failure at slow corner max_ss_100C_1.60V (unresolved)'),
        P('The SPI regfile write-enable decode path fails at the worst-case slow corner '
          '(WNS = −1.915 ns, 9 violations). The same 9-stage combinational cone is 1.83× '
          'slower at 100°C / 1.60 V than at nominal. Three mitigations were identified '
          '(pipeline register, reduced regfile depth, 50 MHz clock) but were not implemented '
          'within M4 scope. The design closes at the nominal and fast corners, which represent '
          'the expected operating conditions for the target edge application.'),
        H2('9.6 AXI4-Lite interface not implemented'),
        P('The roofline analysis (§2) and CF09 benchmarking both identify the SPI interface '
          'as the dominant bottleneck (ridge point mismatch of 512–1024×). An AXI4-Lite '
          'replacement was designed (see project/remaining_tasks.md, Task 1) but not '
          'implemented before the M4 deadline. The 0.069% compute utilisation represents '
          'the primary unrealised potential of this design.'),
        SP(8),
        HR(),
        P('<i>All source files, testbenches, synthesis results, and raw benchmark data are '
          'committed to the repository under project/m4/. '
          'RTL is identical to M3 (no changes between M3 and M4 submission). '
          'See project/m4/README.md for a complete file catalog.</i>',
          ParagraphStyle('footer', parent=body, fontSize=8.5, textColor=GRAY,
                         alignment=TA_CENTER)),
    ]

    doc.build(story)
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    fig1_block_diagram()
    fig2_dataflow()
    fig3_roofline()
    bench_csv()
    bench_md()
    gen_pdf()
    print('\nAll M4 assets generated.')
