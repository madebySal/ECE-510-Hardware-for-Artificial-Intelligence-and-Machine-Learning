`timescale 1ns/1ps
// Top-level integration: SPI interface + BNN compute core
//
// Data flow:
//   1. Host writes ACT_BYTES activation bytes over SPI (addr 0x00..ACT_BYTES-1)
//   2. Host writes WGT_BYTES weight bytes over SPI (addr ACT_BYTES..ACT_BYTES+WGT_BYTES-1)
//   3. Host writes 0x01 to CTRL (addr 0x7E) → compute_start pulse
//   4. compute_core runs XNOR+popcount in one clock cycle → result_valid goes high
//   5. Host polls STATUS (addr 0x7F): bit1=result_valid, bit0=result_out
//
// Parameters:
//   N -- neuron input width in bits (default 64 for sim; 784 for production)
//
// External ports: clk, rst, 4 SPI pins only.

module bnn_top #(
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

    // Internal wiring between interface and compute core
    logic [N-1:0] act_vec;        // activation vector from SPI to core
    logic [N-1:0] wgt_vec;        // weight vector from SPI to core
    logic         compute_start;  // one-cycle trigger from CTRL register
    logic         result_out;     // 1-bit neuron result (core → SPI status)
    logic         result_valid;   // result stable flag (core → SPI status)

    // SPI slave / register file
    \interface #(.N(N)) u_if (
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

    // XNOR + popcount compute core
    compute_core #(.N(N)) u_core (
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
