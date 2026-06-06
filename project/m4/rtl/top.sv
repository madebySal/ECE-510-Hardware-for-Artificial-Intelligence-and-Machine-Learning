`timescale 1ns/1ps
// M3 Top-level integration — BNN Inference Accelerator
// =====================================================
// Instantiates the M2 interface and compute_core modules and wires them
// together. This is the design-under-synthesis for OpenLane 2.
//
// External ports:
//   clk   -- input,  1b,    100 MHz system clock
//   rst   -- input,  1b,    synchronous active-high reset
//   sck   -- input,  1b,    SPI clock from host (Mode 0, CPOL=0 CPHA=0)
//   cs_n  -- input,  1b,    SPI chip-select, active-low
//   mosi  -- input,  1b,    SPI master-out slave-in
//   miso  -- output, 1b,    SPI master-in slave-out
//
// Parameters:
//   N     -- neuron input width in bits (default 64 for simulation; 784 for production)
//
// Data flow (end-to-end):
//   1. Host writes ACT_BYTES activation bytes via SPI burst (addr 0x00..0x07)
//   2. Host writes WGT_BYTES weight bytes via SPI burst (addr 0x08..0x0F)
//   3. Host writes 0x01 to CTRL register (addr 0x7E) → compute_start pulse
//   4. compute_core performs XNOR+popcount in one clock cycle
//   5. Host polls STATUS register (addr 0x7F): bit1=result_valid, bit0=result_out
//
// Glue logic: none required.
//   The interface's compute_start output connects directly to both act_valid and
//   weight_valid inputs of the compute core. The act_out / wgt_out buses connect
//   directly to activation / weight_row. The result_out and result_valid feedback
//   signals connect directly back. No FIFOs, no clock-domain crossings, no width
//   converters — both modules share a single clock domain (clk).
//
// Synthesis target: SKY130A via OpenLane 2 (see project/m3/synth/config.json).
// Simulator: Icarus Verilog 12.0 (see project/m3/README.md for run commands).

module top #(
    parameter int N = 64
) (
    input  logic clk,
    input  logic rst,

    // SPI pins (to/from host MCU)
    input  logic sck,
    input  logic cs_n,
    input  logic mosi,
    output logic miso
);

    // -----------------------------------------------------------------------
    // Internal signals connecting interface ↔ compute_core
    // -----------------------------------------------------------------------
    logic [N-1:0] act_vec;        // activation vector: interface → core
    logic [N-1:0] wgt_vec;        // weight vector:     interface → core
    logic         compute_start;  // one-cycle trigger:  interface → core (both valid signals)
    logic         result_out;     // neuron result bit:  core → interface
    logic         result_valid;   // result stable flag: core → interface

    // -----------------------------------------------------------------------
    // M2 SPI slave / register file
    // -----------------------------------------------------------------------
    \interface #(
        .N (N)
    ) u_interface (
        .clk           (clk),
        .rst           (rst),
        .sck           (sck),
        .cs_n          (cs_n),
        .mosi          (mosi),
        .miso          (miso),
        .act_out       (act_vec),
        .wgt_out       (wgt_vec),
        .compute_start (compute_start),
        .result_out    (result_out),
        .result_valid  (result_valid)
    );

    // -----------------------------------------------------------------------
    // M2 XNOR + popcount compute core
    // -----------------------------------------------------------------------
    compute_core #(
        .N (N)
    ) u_compute_core (
        .clk          (clk),
        .rst          (rst),
        .activation   (act_vec),
        .act_valid    (compute_start),
        .weight_row   (wgt_vec),
        .weight_valid (compute_start),
        .out          (result_out),
        .result_valid (result_valid)
    );

endmodule
