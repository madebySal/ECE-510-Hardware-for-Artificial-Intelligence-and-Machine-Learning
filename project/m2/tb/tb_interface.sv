`timescale 1ns/1ps
// Testbench: interface_module (SPI Mode-0 slave, CPOL=0 CPHA=0)
// Simulator: Icarus Verilog  iverilog -g2012 tb_interface.sv ../rtl/interface.sv && ./a.out
// Tests
//   1. Write transaction  — burst-write 4 bytes, read back; verify stored values
//   2. Activation unpack  — write 8 activation bytes; verify act_out assembly
//   3. Status register    — drive result_out/valid; read STATUS reg; verify bits

module tb_interface;

    localparam int N         = 64;
    localparam int ACT_BYTES = (N+7)/8;   // 8
    localparam realtime CLK_HALF = 5ns;
    localparam realtime SCK_HALF = 40ns;

    logic        clk = 0;
    logic        rst;
    logic        sck = 0;
    logic        cs_n = 1;
    logic        mosi = 0;
    logic        miso;
    logic [N-1:0] act_out;
    logic [N-1:0] wgt_out;
    logic         compute_start;
    logic         result_out  = 0;
    logic         result_valid = 0;

    always #(CLK_HALF) clk = ~clk;

    interface_module #(.N(N)) dut (
        .clk          (clk),
        .rst          (rst),
        .sck          (sck),
        .cs_n         (cs_n),
        .mosi         (mosi),
        .miso         (miso),
        .act_out      (act_out),
        .wgt_out      (wgt_out),
        .compute_start(compute_start),
        .result_out   (result_out),
        .result_valid (result_valid)
    );

    int pass_cnt = 0;
    int fail_cnt = 0;

    // Shared SPI byte I/O
    reg [7:0] spi_tx_byte;
    reg [7:0] spi_rx_byte;

    // Shared storage for burst (max 16 bytes)
    reg [7:0] buf_wr [0:15];
    reg [7:0] buf_rd [0:15];

    // Send/receive one SPI byte
    task spi_byte_xfer;
        integer b;
        spi_rx_byte = 8'h00;
        for (b = 7; b >= 0; b = b - 1) begin
            mosi = spi_tx_byte[b];
            #(SCK_HALF); sck = 1;
            #(SCK_HALF); spi_rx_byte = {spi_rx_byte[6:0], miso}; sck = 0;
            #(SCK_HALF);
        end
    endtask

    // Write burst: addr, pulls bytes from buf_wr[0..n-1]
    task spi_write_burst;
        input [6:0] addr;
        input integer n;
        integer i;
        cs_n = 0; #(SCK_HALF);
        spi_tx_byte = {1'b1, addr}; spi_byte_xfer;   // CMD
        for (i = 0; i < n; i = i + 1) begin
            spi_tx_byte = buf_wr[i]; spi_byte_xfer;
        end
        cs_n = 1; mosi = 0; #(SCK_HALF*4);
    endtask

    // Read burst: addr, n bytes, stores result in buf_rd[0..n-1]
    task spi_read_burst;
        input [6:0] addr;
        input integer n;
        integer i;
        cs_n = 0; #(SCK_HALF);
        spi_tx_byte = {1'b0, addr}; spi_byte_xfer;   // CMD
        for (i = 0; i < n; i = i + 1) begin
            spi_tx_byte = 8'h00; spi_byte_xfer;
            buf_rd[i] = spi_rx_byte;
        end
        cs_n = 1; mosi = 0; #(SCK_HALF*4);
    endtask

    task hw_reset;
        rst = 1; sck = 0; cs_n = 1; mosi = 0;
        result_out = 0; result_valid = 0;
        repeat(6) @(posedge clk);
        rst = 0;
        @(posedge clk);
    endtask

    integer i;
    reg [63:0] act_val;

    initial begin
        $display("=== tb_interface start ===");
        hw_reset();

        // ---- Test 1: Write 4 bytes, read back ----
        buf_wr[0]=8'hAB; buf_wr[1]=8'hCD; buf_wr[2]=8'hEF; buf_wr[3]=8'h12;
        spi_write_burst(7'h00, 4);
        spi_read_burst (7'h00, 4);
        if (buf_rd[0]===8'hAB && buf_rd[1]===8'hCD &&
            buf_rd[2]===8'hEF && buf_rd[3]===8'h12) begin
            $display("  PASS  write_read_back: %02h %02h %02h %02h",
                     buf_rd[0], buf_rd[1], buf_rd[2], buf_rd[3]);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL  write_read_back: got %02h %02h %02h %02h",
                     buf_rd[0], buf_rd[1], buf_rd[2], buf_rd[3]);
            fail_cnt = fail_cnt + 1;
        end

        hw_reset();

        // ---- Test 2: Activation unpack ----
        act_val = 64'hDEAD_BEEF_CAFE_1234;
        for (i = 0; i < ACT_BYTES; i = i + 1)
            buf_wr[i] = act_val[i*8 +: 8];
        spi_write_burst(7'h00, ACT_BYTES);
        repeat(4) @(posedge clk);
        if (act_out === act_val) begin
            $display("  PASS  act_unpack: act_out=0x%016h", act_out);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL  act_unpack: got=0x%016h expected=0x%016h", act_out, act_val);
            fail_cnt = fail_cnt + 1;
        end

        hw_reset();

        // ---- Test 3: STATUS register ----
        result_out   = 1'b1;
        result_valid = 1'b1;
        repeat(4) @(posedge clk);
        spi_read_burst(7'h7F, 1);
        if ((buf_rd[0] & 8'h03) === 8'h03) begin
            $display("  PASS  status_reg: STATUS=0x%02h", buf_rd[0]);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL  status_reg: STATUS=0x%02h (expected bits[1:0]=11)", buf_rd[0]);
            fail_cnt = fail_cnt + 1;
        end

        $display("=== Results: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("PASS");
        else
            $display("FAIL");

        $finish;
    end

endmodule
