`timescale 1ns/1ps
// Compute core: BNN fully-connected layer — XNOR-popcount
// Computes one output neuron: popcount(XNOR(activation, weight_row))
// then applies binary threshold: out = (pop > N/2) ? +1 : -1
//
// Parameters
//   N            -- input width in bits (default 64 for sim; 784 for production)
//
// Ports
//   clk          -- input,  1b,  system clock
//   rst          -- input,  1b,  synchronous active-high reset
//   activation   -- input,  Nb,  packed binary activation row
//   act_valid    -- input,  1b,  activation data valid strobe
//   weight_row   -- input,  Nb,  packed binary weight row
//   weight_valid -- input,  1b,  weight data valid strobe
//   out          -- output, 1b,  neuron output: 1=+1, 0=-1
//   result_valid -- output, 1b,  pulses high one cycle after data_ready
//
// Clock domain: single clock (clk). No crossings.
// Reset: synchronous, active-high.

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
    logic [ACC_W-1:0]  pop;
    logic              data_ready;

    assign xnor_vec   = ~(activation ^ weight_row);
    assign data_ready = act_valid & weight_valid;

    function automatic logic [ACC_W-1:0] popcount_f(input logic [N-1:0] v);
        automatic int i;
        automatic logic [ACC_W-1:0] acc = '0;
        for (i = 0; i < N; i++) acc = acc + {{(ACC_W-1){1'b0}}, v[i]};
        return acc;
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            out          <= 1'b0;
            result_valid <= 1'b0;
            pop          <= '0;
        end else if (data_ready) begin
            pop          <= popcount_f(xnor_vec);
            out          <= (popcount_f(xnor_vec) > (N / 2)) ? 1'b1 : 1'b0;
            result_valid <= 1'b1;
        end
    end

endmodule
