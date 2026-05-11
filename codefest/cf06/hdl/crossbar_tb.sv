// Testbench for crossbar_mac.sv
// Weight matrix: W[row][col] = [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]
// Input:  in = [10, 20, 30, 40]
//
// Hand-calculated expected outputs (out[j] = sum_i W[i][j]*in[i]):
//   out[0] = +1(10)+1(20)-1(30)-1(40) = -40
//   out[1] = -1(10)+1(20)+1(30)-1(40) =   0
//   out[2] = +1(10)-1(20)+1(30)-1(40) = -20
//   out[3] = -1(10)-1(20)-1(30)+1(40) = -20

`timescale 1ns/1ps

module crossbar_tb;

    logic              clk, rst;
    logic              weight_we;
    logic [1:0]        weight_row, weight_col;
    logic              weight_val;
    logic signed [7:0] in0, in1, in2, in3;
    logic              in_valid;
    logic signed [15:0] out0, out1, out2, out3;
    logic              out_valid;

    crossbar_mac dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    // Task: load one weight into the array
    task load_weight(input int r, input int c, input logic v);
        @(negedge clk);
        weight_we  = 1;
        weight_row = r[1:0];
        weight_col = c[1:0];
        weight_val = v;
        @(posedge clk); #1;
        weight_we = 0;
    endtask

    // Task: apply inputs for one cycle and capture results
    task apply_and_check(
        input signed [7:0]  i0, i1, i2, i3,
        input signed [15:0] e0, e1, e2, e3,
        input string         label
    );
        // Drive inputs before clock edge
        @(negedge clk);
        in0 = i0; in1 = i1; in2 = i2; in3 = i3;
        in_valid = 1;
        // Clock edge: DUT registers and produces output
        @(posedge clk); #1;
        in_valid = 0;
        // Output is now valid — check immediately
        if (out_valid && out0===e0 && out1===e1 && out2===e2 && out3===e3)
            $display("PASS  %s: out=[%0d, %0d, %0d, %0d]",
                     label, out0, out1, out2, out3);
        else begin
            $display("FAIL  %s: got=[%0d, %0d, %0d, %0d]  expected=[%0d, %0d, %0d, %0d]  out_valid=%0b",
                     label, out0, out1, out2, out3, e0, e1, e2, e3, out_valid);
            $fatal(1, "Test failed");
        end
    endtask

    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);

        // Reset
        rst = 1; weight_we = 0; in_valid = 0;
        in0 = 0; in1 = 0; in2 = 0; in3 = 0;
        repeat(3) @(posedge clk); #1;
        rst = 0;

        // ----------------------------------------------------------------
        // Load weight matrix W = [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]
        //   row\col  0   1   2   3
        //     0     +1  -1  +1  -1
        //     1     +1  +1  -1  -1
        //     2     -1  +1  +1  -1
        //     3     -1  -1  -1  +1
        // ----------------------------------------------------------------
        load_weight(0, 0, 1); load_weight(0, 1, 0);
        load_weight(0, 2, 1); load_weight(0, 3, 0);

        load_weight(1, 0, 1); load_weight(1, 1, 1);
        load_weight(1, 2, 0); load_weight(1, 3, 0);

        load_weight(2, 0, 0); load_weight(2, 1, 1);
        load_weight(2, 2, 1); load_weight(2, 3, 0);

        load_weight(3, 0, 0); load_weight(3, 1, 0);
        load_weight(3, 2, 0); load_weight(3, 3, 1);

        // ----------------------------------------------------------------
        // Test 1: in=[10,20,30,40]  expected out=[-40, 0, -20, -20]
        // ----------------------------------------------------------------
        apply_and_check(10, 20, 30, 40,  -40, 0, -20, -20,  "in=[10,20,30,40]");

        // ----------------------------------------------------------------
        // Test 2: all-zero input -> all outputs zero
        // ----------------------------------------------------------------
        apply_and_check(0, 0, 0, 0,  0, 0, 0, 0,  "in=[0,0,0,0]");

        // ----------------------------------------------------------------
        // Test 3: only in[0]=1 active -> out[j] = W[0][j] = [+1,-1,+1,-1]
        // ----------------------------------------------------------------
        apply_and_check(1, 0, 0, 0,  1, -1, 1, -1,  "in=[1,0,0,0]");

        $display("All tests PASS");
        #20 $finish;
    end

endmodule
