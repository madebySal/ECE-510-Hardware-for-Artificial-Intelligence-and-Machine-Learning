`timescale 1ns/1ps
// Compute core: BNN fully-connected layer — XNOR-popcount
// Synthesis version: popcount via combinational always block (no automatic vars)
// N=64 fixed for synthesis target

module compute_core #(
    parameter int N = 64
) (
    input  logic              clk,
    input  logic              rst,

    input  logic [N-1:0]      activation,
    input  logic              act_valid,

    input  logic [N-1:0]      weight_row,
    input  logic              weight_valid,

    output logic              out,
    output logic              result_valid
);

    localparam int ACC_W = $clog2(N) + 1;

    logic [N-1:0]      xnor_vec;
    logic [ACC_W-1:0]  pop_comb;
    logic              data_ready;

    assign xnor_vec   = ~(activation ^ weight_row);
    assign data_ready = act_valid & weight_valid;

    always_comb begin
        pop_comb = '0;
        for (int i = 0; i < N; i++)
            pop_comb = pop_comb + {{(ACC_W-1){1'b0}}, xnor_vec[i]};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            out          <= 1'b0;
            result_valid <= 1'b0;
        end else if (data_ready) begin
            out          <= (pop_comb > ACC_W'(N / 2)) ? 1'b1 : 1'b0;
            result_valid <= 1'b1;
        end else begin
            result_valid <= 1'b0;
        end
    end

endmodule
