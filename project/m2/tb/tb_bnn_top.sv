`timescale 1ns/1ps
// End-to-end testbench for bnn_top.sv
// Drives SPI transactions from the host side; verifies result via STATUS register.
//
// Test cases:
//   1. all_agree:    activation == weight_row → XNOR all-1s → pop=N → out=1
//   2. all_disagree: activation == ~weight_row → XNOR all-0s → pop=0 → out=0
//   3. mixed:        random-ish pattern; expected output verified by reference model
//
// SPI burst protocol:
//   [W/R | ADDR[6:0]] [DATA_BYTE_0] [DATA_BYTE_1] ... → address auto-increments
//   CS deasserted after all bytes to commit the transaction.

module tb_bnn_top;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int N         = 64;
    localparam int ACT_BYTES = (N + 7) / 8;   // 8
    localparam int WGT_BYTES = (N + 7) / 8;   // 8
    localparam int ACT_BASE  = 0;
    localparam int WGT_BASE  = ACT_BYTES;      // 8
    localparam int SCK_HALF  = 40;             // ns (12.5 MHz SPI)

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic clk, rst;
    logic sck, cs_n, mosi, miso;

    bnn_top #(.N(N)) dut (
        .clk  (clk),
        .rst  (rst),
        .sck  (sck),
        .cs_n (cs_n),
        .mosi (mosi),
        .miso (miso)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;   // 100 MHz system clock

    // -----------------------------------------------------------------------
    // Test accounting
    // -----------------------------------------------------------------------
    int pass_cnt = 0;
    int fail_cnt = 0;

    // -----------------------------------------------------------------------
    // Reference model: returns expected binary output
    // -----------------------------------------------------------------------
    function automatic logic ref_bnn(input logic [N-1:0] act, wgt);
        logic [N-1:0] xnor_vec;
        int pop;
        xnor_vec = ~(act ^ wgt);
        pop = 0;
        for (int i = 0; i < N; i++) pop += xnor_vec[i];
        return (pop > N / 2) ? 1'b1 : 1'b0;
    endfunction

    // -----------------------------------------------------------------------
    // SPI tasks
    // -----------------------------------------------------------------------

    // Send/receive one byte; returns received byte
    task automatic spi_byte(input logic [7:0] tx, output logic [7:0] rx);
        rx = 8'b0;
        for (int b = 7; b >= 0; b--) begin
            mosi = tx[b];
            #(SCK_HALF);
            sck = 1;
            #(SCK_HALF);
            rx[b] = miso;
            sck = 0;
            #(SCK_HALF);
        end
    endtask

    // Burst write: assert CS, send CMD byte, then data_bytes[], deassert CS
    task automatic spi_write(
        input  int            addr,
        input  logic [7:0]    data [],
        input  int            n_bytes
    );
        logic [7:0] dummy;
        cs_n = 0;
        #(SCK_HALF);
        spi_byte(8'(8'h80 | (addr & 7'h7F)), dummy);  // CMD: write flag + addr
        for (int i = 0; i < n_bytes; i++)
            spi_byte(data[i], dummy);
        cs_n = 1;
        mosi = 0;
        #(SCK_HALF * 4);
    endtask

    // Single-byte read; returns the received data byte
    task automatic spi_read(input int addr, output logic [7:0] rx_data);
        logic [7:0] dummy;
        cs_n = 0;
        #(SCK_HALF);
        spi_byte(8'(addr & 7'h7F), dummy);    // CMD: read flag cleared + addr
        spi_byte(8'h00, rx_data);             // dummy byte → captures MISO
        cs_n = 1;
        mosi = 0;
        #(SCK_HALF * 4);
    endtask

    // -----------------------------------------------------------------------
    // Load activation + weight over SPI and trigger compute; return result
    // -----------------------------------------------------------------------
    task automatic run_inference(
        input  logic [N-1:0] act,
        input  logic [N-1:0] wgt,
        output logic          result
    );
        logic [7:0] act_bytes [0:7];
        logic [7:0] wgt_bytes [0:7];
        logic [7:0] status;
        int timeout;

        // Pack activation and weight into byte arrays
        for (int i = 0; i < ACT_BYTES; i++)
            act_bytes[i] = act[i*8 +: 8];
        for (int i = 0; i < WGT_BYTES; i++)
            wgt_bytes[i] = wgt[i*8 +: 8];

        // Burst-write activation bytes starting at ACT_BASE (0x00)
        spi_write(ACT_BASE, act_bytes, ACT_BYTES);

        // Burst-write weight bytes starting at WGT_BASE (0x08)
        spi_write(WGT_BASE, wgt_bytes, WGT_BYTES);

        // Write CTRL = 0x01 to trigger compute (addr 0x7E)
        begin
            logic [7:0] ctrl_byte [0:0];
            ctrl_byte[0] = 8'h01;
            spi_write(8'h7E, ctrl_byte, 1);
        end

        // Poll STATUS (0x7F) until result_valid (bit1) set
        timeout = 0;
        status  = 8'h00;
        while (!(status & 8'h02) && timeout < 20) begin
            spi_read(8'h7F, status);
            timeout++;
        end

        if (timeout >= 20) begin
            $display("FAIL  timeout waiting for result_valid");
            fail_cnt++;
            result = 1'bx;
        end else begin
            result = status[0];
        end
    endtask

    // -----------------------------------------------------------------------
    // Stimulus
    // -----------------------------------------------------------------------
    logic [N-1:0] act_vec, wgt_vec;
    logic          got, expected;

    initial begin
        $dumpfile("tb_bnn_top.vcd");
        $dumpvars(0, tb_bnn_top);

        // Initialise SPI pins
        sck  = 0; cs_n = 1; mosi = 0;

        // Reset
        rst = 1;
        repeat (6) @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ----------------------------------------------------------------
        // Test 1: all_agree — activation == weight → pop = N → out = 1
        // ----------------------------------------------------------------
        act_vec  = {N{1'b1}};
        wgt_vec  = {N{1'b1}};
        expected = ref_bnn(act_vec, wgt_vec);   // should be 1
        run_inference(act_vec, wgt_vec, got);

        if (got === expected && got === 1'b1) begin
            $display("PASS  all_agree:    out=%0b (expected %0b)", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  all_agree:    out=%0b (expected %0b)", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 2: all_disagree — activation = all-1, weight = all-0 → pop = 0 → out = 0
        // ----------------------------------------------------------------
        act_vec  = {N{1'b1}};
        wgt_vec  = {N{1'b0}};
        expected = ref_bnn(act_vec, wgt_vec);   // should be 0
        run_inference(act_vec, wgt_vec, got);

        if (got === expected && got === 1'b0) begin
            $display("PASS  all_disagree: out=%0b (expected %0b)", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  all_disagree: out=%0b (expected %0b)", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 3: mixed — alternating bits → pop = N/2, not > N/2 → out = 0
        // ----------------------------------------------------------------
        act_vec  = 64'hAAAA_AAAA_AAAA_AAAA;   // alternating 1010...
        wgt_vec  = 64'hAAAA_AAAA_AAAA_AAAA;   // same → full agreement → pop = N → out = 1
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected) begin
            $display("PASS  mixed:        out=%0b (expected %0b)", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  mixed:        out=%0b (expected %0b)", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 4: half — exactly half agree → pop = N/2, not > N/2 → out = 0
        // ----------------------------------------------------------------
        act_vec  = 64'hFFFF_FFFF_0000_0000;   // high 32 bits = 1, low 32 = 0
        wgt_vec  = 64'hFFFF_FFFF_FFFF_FFFF;   // all 1s → only high half agrees
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected) begin
            $display("PASS  half_agree:   out=%0b (expected %0b)", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  half_agree:   out=%0b (expected %0b)", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Summary
        // ----------------------------------------------------------------
        $display("----------------------------------------");
        if (fail_cnt == 0)
            $display("ALL PASS  (%0d/%0d tests passed)", pass_cnt, pass_cnt + fail_cnt);
        else
            $display("FAILURES  (%0d passed, %0d FAILED)", pass_cnt, fail_cnt);
        $display("----------------------------------------");

        #20 $finish;
    end

endmodule
