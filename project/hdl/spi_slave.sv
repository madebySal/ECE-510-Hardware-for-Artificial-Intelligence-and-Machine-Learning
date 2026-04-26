`timescale 1ns/1ps
// SPI Mode-0 slave — 8-bit register file (system-clock synchronised)
//
// Frame = 16 SCK cycles: [W/R | ADDR[6:0]] [DATA[7:0]]
//   Write: W/R=1 → write DATA to ADDR
//   Read:  W/R=0 → after 8-bit cmd phase, MISO clocks out REG[ADDR]
//
// Registers (5-bit address space):
//   0x00  ACT_LO  — low  byte of activation word (write)
//   0x01  ACT_HI  — high byte; writing ACT_HI latches act_out and pulses act_strobe
//   0x10  RESULT  — compute result byte (read-only)
//   0x11  STATUS  — bit 0 = result_valid (read-only)

module spi_slave (
    input  logic        clk,
    input  logic        rst,

    // SPI pins
    input  logic        sck,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    // Compute-core register interface
    output logic [15:0] act_out,
    output logic        act_strobe,
    input  logic [7:0]  result_in,
    input  logic        result_valid
);

    // -----------------------------------------------------------------------
    // Synchronise SPI pins to system clock (2-FF metastability chain)
    // -----------------------------------------------------------------------
    logic sck_s0,  sck_s1;
    logic cs_s0,   cs_s1;
    logic mosi_s0, mosi_s1;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            {sck_s1,  sck_s0}  <= 2'b0;
            {cs_s1,   cs_s0}   <= 2'b11;
            {mosi_s1, mosi_s0} <= 2'b0;
        end else begin
            sck_s0  <= sck;   sck_s1  <= sck_s0;
            cs_s0   <= cs_n;  cs_s1   <= cs_s0;
            mosi_s0 <= mosi;  mosi_s1 <= mosi_s0;
        end
    end

    wire sck_rise  = ( sck_s0 & ~sck_s1);
    wire sck_fall  = (~sck_s0 &  sck_s1);
    wire cs_active = ~cs_s1;

    // -----------------------------------------------------------------------
    // 8-bit register file
    // -----------------------------------------------------------------------
    logic [7:0] regfile [0:31];

    // -----------------------------------------------------------------------
    // Shift / decode state
    // -----------------------------------------------------------------------
    logic [4:0]  bit_cnt;      // 0..16; 5-bit to detect 16
    logic [15:0] rx_sr;        // receive shift register
    logic [15:0] rx_latch;     // snapshot of rx_sr when CS deasserts
    logic [7:0]  tx_sr;        // transmit shift register
    logic        frame_done;   // pulses for one cycle after CS deasserts

    // Track previous cs_active to detect falling edge
    logic cs_active_prev;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            bit_cnt       <= 5'd0;
            rx_sr         <= 16'b0;
            rx_latch      <= 16'b0;
            tx_sr         <= 8'b0;
            miso          <= 1'b0;
            act_out       <= 16'b0;
            act_strobe    <= 1'b0;
            frame_done    <= 1'b0;
            cs_active_prev <= 1'b0;
            for (int i = 0; i < 32; i++) regfile[i] <= 8'b0;
        end else begin
            act_strobe    <= 1'b0;
            frame_done    <= 1'b0;
            cs_active_prev <= cs_active;

            // Mirror read-only compute signals
            regfile[5'h10] <= result_in;
            regfile[5'h11] <= {7'b0, result_valid};

            // Detect CS deassert (falling edge of cs_active)
            if (cs_active_prev && !cs_active) begin
                if (bit_cnt == 5'd16) begin
                    rx_latch   <= rx_sr;
                    frame_done <= 1'b1;
                end
                bit_cnt <= 5'd0;
                rx_sr   <= 16'b0;
                miso    <= 1'b0;
            end

            if (cs_active) begin
                if (sck_rise) begin
                    rx_sr   <= {rx_sr[14:0], mosi_s1};
                    if (bit_cnt < 5'd16)
                        bit_cnt <= bit_cnt + 5'd1;

                    // After 8th bit received, pre-load TX shift register for reads
                    // At this point rx_sr[6:0] holds bits 7..1 of cmd byte, mosi_s1 is bit 0
                    if (bit_cnt == 5'd7) begin
                        tx_sr <= regfile[{rx_sr[3:0], mosi_s1}];  // ADDR[4:0]
                    end
                end

                if (sck_fall && bit_cnt >= 5'd8) begin
                    miso  <= tx_sr[7];
                    tx_sr <= {tx_sr[6:0], 1'b0};
                end
            end

            // Execute write from latched frame (one cycle after CS deassert)
            if (frame_done) begin
                if (rx_latch[15]) begin   // W/R bit = 1 → write
                    regfile[rx_latch[12:8]] <= rx_latch[7:0];
                    if (rx_latch[12:8] == 5'h01) begin
                        act_out    <= {rx_latch[7:0], regfile[5'h00]};
                        act_strobe <= 1'b1;
                    end
                end
            end
        end
    end

endmodule
