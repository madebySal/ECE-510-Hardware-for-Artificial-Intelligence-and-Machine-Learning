// Testbench for 4-bit adder
module adder4_tb;
    reg  [3:0] a, b;
    wire [3:0] sum;
    wire       carry_out;

    adder4 uut (.a(a), .b(b), .sum(sum), .carry_out(carry_out));

    initial begin
        $display("  A    B  | SUM  CARRY");
        $display("----------------------");

        a = 4'd3;  b = 4'd5;  #10;
        $display("  %0d    %0d  |  %0d     %0d", a, b, sum, carry_out);

        a = 4'd15; b = 4'd1;  #10;
        $display("  %0d   %0d  |  %0d     %0d", a, b, sum, carry_out);

        a = 4'd15; b = 4'd15; #10;
        $display("  %0d   %0d  | %0d    %0d", a, b, sum, carry_out);

        a = 4'd0;  b = 4'd0;  #10;
        $display("  %0d    %0d  |  %0d     %0d", a, b, sum, carry_out);

        $display("----------------------");
        $display("Simulation complete.");
        $finish;
    end
endmodule
