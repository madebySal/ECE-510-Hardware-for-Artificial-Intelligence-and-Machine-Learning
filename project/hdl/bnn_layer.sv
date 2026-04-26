// BNN fully-connected layer: XNOR-popcount compute core
// Computes one output neuron: popcount(XNOR(activation, weight_row))
// then applies binary threshold (sign): out = (pop >= N/2) ? +1 : -1
//
// Parameters
//   N   -- input width in bits (e.g. 784 for layer 1)
//
// Interface
//   SPI host loads weight row (N bits) via spi_data/spi_valid handshake.
//   Activation row (N bits) applied simultaneously.
//   result_valid pulses high for one cycle when out is stable.
//
// Precision choice: 1-bit (binary) weights and activations.
// Popcount accumulator width = $clog2(N)+1 to hold 0..N.

module bnn_layer #(
    parameter int N = 784
) (
    input  logic              clk,
    input  logic              rst,

    // input activation (N bits, binary packed)
    input  logic [N-1:0]      activation,
    input  logic              act_valid,

    // weight row (N bits, preloaded)
    input  logic [N-1:0]      weight_row,
    input  logic              weight_valid,

    // output: +1 encoded as 1, -1 encoded as 0
    output logic              out,
    output logic              result_valid
);

    localparam int ACC_W = $clog2(N) + 1;  // enough bits for 0..N

    logic [N-1:0]      xnor_vec;
    logic [ACC_W-1:0]  pop;
    logic              data_ready;

    // XNOR: positions where activation == weight contribute +1
    assign xnor_vec   = ~(activation ^ weight_row);
    assign data_ready = act_valid & weight_valid;

    // Popcount via generate tree (synthesizable)
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
            // threshold: if popcount > N/2 => more agreements => output +1 (1)
            out          <= (popcount_f(xnor_vec) > (N / 2)) ? 1'b1 : 1'b0;
            result_valid <= 1'b1;
        end else begin
            result_valid <= 1'b0;
        end
    end

endmodule
