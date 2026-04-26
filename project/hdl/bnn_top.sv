`timescale 1ns/1ps
// Top-level integration: SPI slave + BNN compute core
//
// Data flow:
//   Host writes 98 activation bytes (addrs 0x00-0x61) over SPI
//   Host writes 98 weight-row bytes (addrs 0x62-0xC3) over SPI
//   Host writes 0x01 to CTRL reg (addr 0x7E) → compute_start pulse
//   bnn_layer latches activation + weight_row, runs XNOR+popcount in one clock
//   Host polls STATUS reg (addr 0x7F) bit1 for result_valid, reads bit0 for result

module bnn_top #(
    parameter int N = 64    // testbench default; set to 784 for production
) (
    input  logic clk,
    input  logic rst,

    // SPI pins (exposed to top-level pins)
    input  logic sck,
    input  logic cs_n,
    input  logic mosi,
    output logic miso
);

    logic [N-1:0] act_vec;
    logic [N-1:0] wgt_vec;
    logic         compute_start;
    logic         result_out;
    logic         result_valid;

    spi_slave #(.N(N)) u_spi (
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

    bnn_layer #(.N(N)) u_bnn (
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
