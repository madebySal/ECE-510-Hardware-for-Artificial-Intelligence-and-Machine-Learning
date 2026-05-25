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
//   result_valid -- output, 1b,  goes high one cycle after data_ready
//
// Clock domain: single clock (clk). No crossings.
// Reset: synchronous, active-high.
//
// Synthesis note: popcount is implemented via $countones(), which Yosys
// maps to a balanced binary adder tree. The earlier function-based
// implementation was removed because Yosys does not support 'return'
// statements inside synthesisable functions (as of the OpenLane 2.3.10
// bundled Yosys). $countones is the canonical SV popcount primitive and
// is fully synthesisable.

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
    logic [ACC_W-1:0]  pop_comb;   // combinational popcount
    logic              data_ready;

    assign xnor_vec   = ~(activation ^ weight_row);
    assign data_ready = act_valid & weight_valid;

    // $countones: standard SV popcount; Yosys synthesises as adder tree
    assign pop_comb = ACC_W'($countones(xnor_vec));

    always_ff @(posedge clk) begin
        if (rst) begin
            out          <= 1'b0;
            result_valid <= 1'b0;
        end else if (data_ready) begin
            out          <= (pop_comb > ACC_W'(N / 2)) ? 1'b1 : 1'b0;
            result_valid <= 1'b1;
        end
    end

endmodule
