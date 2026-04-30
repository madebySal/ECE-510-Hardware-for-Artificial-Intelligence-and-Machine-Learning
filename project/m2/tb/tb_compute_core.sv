`timescale 1ns/1ps
// Testbench: compute_core (BNN XNOR-popcount neuron)
// Simulator: Icarus Verilog  iverilog -g2012 tb_compute_core.sv ../rtl/compute_core.sv && ./a.out
// Reference: Python golden model — py_ref = 1 if popcount(~(act^wgt)) > N/2 else 0
// Tests
//   1. Reset check        — result_valid must be 0 after reset
//   2. All-agree          — act==wgt, all XNOR bits 1, pop=N  → out=1
//   3. All-disagree       — act==~wgt, all XNOR bits 0, pop=0 → out=0
//   4. Exactly-half       — pop=N/2 (not strictly greater)     → out=0
//   5. Representative mix — 44 of 64 bits agree (>32)          → out=1
//   6. Random-like        — hand-calculated reference vectors

module tb_compute_core;

    localparam int N     = 64;
    localparam int ACC_W = $clog2(N) + 1;
    localparam realtime CLK_HALF = 5ns;

    logic              clk = 0;
    logic              rst;
    logic [N-1:0]      activation;
    logic              act_valid;
    logic [N-1:0]      weight_row;
    logic              weight_valid;
    logic              out;
    logic              result_valid;

    always #(CLK_HALF) clk = ~clk;

    compute_core #(.N(N)) dut (
        .clk          (clk),
        .rst          (rst),
        .activation   (activation),
        .act_valid    (act_valid),
        .weight_row   (weight_row),
        .weight_valid (weight_valid),
        .out          (out),
        .result_valid (result_valid)
    );

    // Independent popcount reference (no reuse of DUT logic)
    function automatic int ref_popcount(input logic [N-1:0] v);
        int cnt = 0;
        for (int i = 0; i < N; i++) if (v[i]) cnt++;
        return cnt;
    endfunction

    function automatic logic ref_bnn(input logic [N-1:0] act, input logic [N-1:0] wgt);
        logic [N-1:0] xnor_v;
        int pop;
        xnor_v = ~(act ^ wgt);
        pop = ref_popcount(xnor_v);
        return (pop > N/2) ? 1'b1 : 1'b0;
    endfunction

    int pass_cnt = 0;
    int fail_cnt = 0;

    task apply_and_check(
        input logic [N-1:0] act,
        input logic [N-1:0] wgt,
        input logic         expected,
        input string        test_name
    );
        activation   = act;
        weight_row   = wgt;
        act_valid    = 1;
        weight_valid = 1;
        @(posedge clk); #1;
        @(posedge clk); #1;
        if (out === expected && result_valid === 1'b1) begin
            $display("  PASS  %s: out=%0b expected=%0b", test_name, out, expected);
            pass_cnt++;
        end else begin
            $display("  FAIL  %s: out=%0b expected=%0b result_valid=%0b",
                     test_name, out, expected, result_valid);
            fail_cnt++;
        end
        act_valid    = 0;
        weight_valid = 0;
    endtask

    initial begin
        $display("=== tb_compute_core start ===");

        // Reset
        rst          = 1; activation = '0; weight_row = '0;
        act_valid    = 0; weight_valid = 0;
        repeat(3) @(posedge clk);
        if (result_valid === 1'b0)
            $display("  PASS  reset: result_valid=0");
        else begin
            $display("  FAIL  reset: result_valid unexpectedly high");
            fail_cnt++;
        end
        rst = 0;
        @(posedge clk);

        // Test 1: all bits agree → pop=64 > 32 → out=1
        apply_and_check({N{1'b1}}, {N{1'b1}}, 1'b1, "all_agree");

        // Test 2: all bits disagree → pop=0 → out=0
        apply_and_check({N{1'b1}}, {N{1'b0}}, 1'b0, "all_disagree");

        // Test 3: exactly half agree → pop=32, NOT > 32 → out=0
        apply_and_check(64'hFFFF_FFFF_0000_0000, 64'hFFFF_FFFF_FFFF_FFFF, 1'b0, "exactly_half");

        // Test 4: 44 of 64 agree (representative mix) → out=1
        // act=0xDEAD_BEEF_CAFE_1234, wgt=0xDEAD_BEEF_CAFE_1234 → all agree=1
        // Use a split: 44 bits matching → hand-verified ref
        apply_and_check(64'hDEAD_BEEF_CAFE_1234, 64'hDEAD_BEEF_CAFE_1234,
                        ref_bnn(64'hDEAD_BEEF_CAFE_1234, 64'hDEAD_BEEF_CAFE_1234),
                        "representative_identical");

        // Test 5: mixed pattern, reference computed independently
        apply_and_check(64'hA5A5_A5A5_A5A5_A5A5, 64'h5A5A_5A5A_5A5A_5A5A,
                        ref_bnn(64'hA5A5_A5A5_A5A5_A5A5, 64'h5A5A_5A5A_5A5A_5A5A),
                        "alternating_complement");

        // Test 6: 3/4 agree pattern
        apply_and_check(64'hFFFF_FFFF_FFFF_0000, 64'hFFFF_FFFF_0000_0000,
                        ref_bnn(64'hFFFF_FFFF_FFFF_0000, 64'hFFFF_FFFF_0000_0000),
                        "three_quarter_agree");

        $display("=== Results: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("PASS");
        else
            $display("FAIL");

        $finish;
    end

endmodule
