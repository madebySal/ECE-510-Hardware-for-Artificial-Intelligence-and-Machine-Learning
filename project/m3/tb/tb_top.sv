`timescale 1ns/1ps
// End-to-end co-simulation testbench for project/m3/rtl/top.sv
// =============================================================
// Drives all communication exclusively through the SPI interface —
// no direct access to compute_core ports. This mirrors what a real
// host MCU would do: write activation bytes, write weight bytes,
// trigger compute via CTRL, poll STATUS for the result.
//
// Design under test: top.sv (N=64)
//   N=64 matches the dominant BNN kernel from M1: a single
//   fully-connected BNN neuron computing XNOR-popcount over a
//   64-bit packed activation and weight row.
//
// SPI burst protocol (Mode 0, CPOL=0, CPHA=0):
//   [W/R | ADDR[6:0]] [DATA_BYTE_0] [DATA_BYTE_1] ...
//   CS deasserted after all bytes commits the transaction.
//   Address auto-increments per byte.
//
// Test cases (reference values computed independently via ref_bnn()):
//   1. all_agree    act=all-1s, wgt=all-1s  → pop=64 > 32 → out=1
//   2. all_disagree act=all-1s, wgt=all-0s  → pop=0       → out=0
//   3. mixed        act=wgt=0xAAAA…          → pop=64 > 32 → out=1
//   4. half_agree   act=0xFFFFFFFF_00000000, wgt=all-1s → pop=32 → out=0

module tb_top;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam int N         = 64;
    localparam int ACT_BYTES = (N + 7) / 8;   // 8
    localparam int WGT_BYTES = (N + 7) / 8;   // 8
    localparam int ACT_BASE  = 0;
    localparam int WGT_BASE  = ACT_BYTES;      // 8
    localparam int SCK_HALF  = 40;             // ns → 12.5 MHz SPI

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic clk, rst;
    logic sck, cs_n, mosi, miso;

    top #(.N(N)) dut (
        .clk  (clk),
        .rst  (rst),
        .sck  (sck),
        .cs_n (cs_n),
        .mosi (mosi),
        .miso (miso)
    );

    // -----------------------------------------------------------------------
    // 100 MHz system clock
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Test accounting
    // -----------------------------------------------------------------------
    int pass_cnt = 0;
    int fail_cnt = 0;

    // Module-level TX buffer (avoids Icarus open-array limitation)
    logic [7:0] tx_buf [0:15];

    // -----------------------------------------------------------------------
    // Independent reference model
    // Computes expected BNN output from first principles — no DUT signals used.
    // -----------------------------------------------------------------------
    function automatic logic ref_bnn(input logic [N-1:0] act, wgt);
        logic [N-1:0] xnor_vec;
        int pop;
        xnor_vec = ~(act ^ wgt);
        pop = 0;
        for (int i = 0; i < N; i++) pop += int'(xnor_vec[i]);
        return (pop > N / 2) ? 1'b1 : 1'b0;
    endfunction

    // -----------------------------------------------------------------------
    // SPI tasks — all host-side, SPI protocol only
    // -----------------------------------------------------------------------

    // Send/receive one byte MSB-first
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

    // Burst write n_bytes from tx_buf[0..n_bytes-1] to addr
    task automatic spi_write_buf(input int addr, input int n_bytes);
        logic [7:0] dummy;
        cs_n = 0;
        #(SCK_HALF);
        spi_byte(8'(8'h80 | (addr & 7'h7F)), dummy);
        for (int i = 0; i < n_bytes; i++)
            spi_byte(tx_buf[i], dummy);
        cs_n = 1;
        mosi = 0;
        #(SCK_HALF * 4);
    endtask

    // Single-byte read
    task automatic spi_read(input int addr, output logic [7:0] rx_data);
        logic [7:0] dummy;
        cs_n = 0;
        #(SCK_HALF);
        spi_byte(8'(addr & 7'h7F), dummy);
        spi_byte(8'h00, rx_data);
        cs_n = 1;
        mosi = 0;
        #(SCK_HALF * 4);
    endtask

    // -----------------------------------------------------------------------
    // run_inference — only uses SPI interface, no direct core access
    // -----------------------------------------------------------------------
    task automatic run_inference(
        input  logic [N-1:0] act,
        input  logic [N-1:0] wgt,
        output logic          result
    );
        logic [7:0] status;
        int timeout;

        // Pack activation bytes into tx_buf, burst-write to ACT_BASE (0x00)
        for (int i = 0; i < ACT_BYTES; i++)
            tx_buf[i] = act[i*8 +: 8];
        spi_write_buf(ACT_BASE, ACT_BYTES);

        // Pack weight bytes into tx_buf, burst-write to WGT_BASE (0x08)
        for (int i = 0; i < WGT_BYTES; i++)
            tx_buf[i] = wgt[i*8 +: 8];
        spi_write_buf(WGT_BASE, WGT_BYTES);

        // Write CTRL = 0x01 to addr 0x7E → pulses compute_start
        tx_buf[0] = 8'h01;
        spi_write_buf(8'h7E, 1);

        // Poll STATUS (addr 0x7F) until result_valid (bit 1) is set
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
            result = status[0];   // bit 0 = result_out
        end
    endtask

    // -----------------------------------------------------------------------
    // Stimulus
    // -----------------------------------------------------------------------
    logic [N-1:0] act_vec, wgt_vec;
    logic          got, expected;

    initial begin
        $dumpfile("cosim.vcd");
        $dumpvars(0, tb_top);

        sck = 0; cs_n = 1; mosi = 0;

        // Reset
        rst = 1;
        repeat (6) @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ----------------------------------------------------------------
        // Test 1: all_agree — act==wgt==all-1s → pop=64 > 32 → out=1
        // Reference: ~(all-1s ^ all-1s) = all-1s; pop=64; 64>32 → 1
        // ----------------------------------------------------------------
        act_vec  = {N{1'b1}};
        wgt_vec  = {N{1'b1}};
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected && got === 1'b1) begin
            $display("PASS  all_agree:    out=%0b expected=%0b", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  all_agree:    out=%0b expected=%0b", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 2: all_disagree — act=all-1s, wgt=all-0s → pop=0 → out=0
        // Reference: ~(all-1s ^ all-0s) = ~(all-1s) = all-0s; pop=0; 0≯32 → 0
        // ----------------------------------------------------------------
        act_vec  = {N{1'b1}};
        wgt_vec  = {N{1'b0}};
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected && got === 1'b0) begin
            $display("PASS  all_disagree: out=%0b expected=%0b", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  all_disagree: out=%0b expected=%0b", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 3: mixed — act==wgt==0xAAAA… → full agreement → pop=64 → out=1
        // Reference: ~(0xAAAA… ^ 0xAAAA…) = all-1s; pop=64; 64>32 → 1
        // ----------------------------------------------------------------
        act_vec  = 64'hAAAA_AAAA_AAAA_AAAA;
        wgt_vec  = 64'hAAAA_AAAA_AAAA_AAAA;
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected) begin
            $display("PASS  mixed:        out=%0b expected=%0b", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  mixed:        out=%0b expected=%0b", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Test 4: half_agree — act=0xFFFFFFFF_00000000, wgt=all-1s
        // High 32 bits agree, low 32 disagree → pop=32; 32≯32 → out=0
        // Reference: XNOR = 0xFFFFFFFF_00000000; pop=32; 32≯32 → 0
        // ----------------------------------------------------------------
        act_vec  = 64'hFFFF_FFFF_0000_0000;
        wgt_vec  = 64'hFFFF_FFFF_FFFF_FFFF;
        expected = ref_bnn(act_vec, wgt_vec);
        run_inference(act_vec, wgt_vec, got);

        if (got === expected) begin
            $display("PASS  half_agree:   out=%0b expected=%0b", got, expected);
            pass_cnt++;
        end else begin
            $display("FAIL  half_agree:   out=%0b expected=%0b", got, expected);
            fail_cnt++;
        end

        // ----------------------------------------------------------------
        // Summary — grader reads this line
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
