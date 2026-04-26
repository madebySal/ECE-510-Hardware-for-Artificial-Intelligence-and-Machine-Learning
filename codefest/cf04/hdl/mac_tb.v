`timescale 1ns/1ps

module mac_tb;
    logic        clk;
    logic        rst;
    logic signed [7:0]  a, b;
    logic signed [31:0] out;

    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .out(out));

    initial clk = 0;
    always #5 clk = ~clk;

    task check(input signed [31:0] expected, input string label);
        @(posedge clk); #1;
        if (out !== expected)
            $display("FAIL %s: got %0d expected %0d", label, out, expected);
        else
            $display("PASS %s: out=%0d", label, out);
    endtask

    initial begin
        $dumpfile("mac_tb.vcd");
        $dumpvars(0, mac_tb);

        // Reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        rst = 0;

        // [a=3, b=4] for 3 cycles — expect 12, 24, 36
        a = 3; b = 4;
        check(12,  "cyc1");
        check(24,  "cyc2");
        check(36,  "cyc3");

        // Assert reset
        rst = 1;
        @(posedge clk); #1;
        if (out !== 0) $display("FAIL reset: out=%0d", out);
        else           $display("PASS reset: out=0");
        rst = 0;

        // [a=-5, b=2] for 2 cycles — expect -10, -20
        a = -5; b = 2;
        check(-10, "neg_cyc1");
        check(-20, "neg_cyc2");

        // Reset again
        rst = 1; @(posedge clk); #1; rst = 0;

        // Large values: a=100, b=100 — product=10000, exceeds 8-bit range
        // Correct: 10000, 20000, 30000
        // Buggy (8-bit truncation): 100*100 mod 256 = 16 → 16, 32, 48
        a = 100; b = 100;
        check(10000, "large_cyc1");
        check(20000, "large_cyc2");
        check(30000, "large_cyc3");

        $display("Testbench complete");
        $finish;
    end
endmodule
