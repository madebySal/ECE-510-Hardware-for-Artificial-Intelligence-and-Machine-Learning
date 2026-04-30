`timescale 1ns/1ps
// Interface module: SPI Mode-0 slave — burst register file for BNN inference control
// Protocol: MSB-first, CS active-low
//   Byte 0:    [W/R | ADDR[6:0]]  command byte; sets base address
//   Bytes 1..: data bytes (write→stored; read→clocked out on MISO)
//              address auto-increments per byte (burst transfer)
//   CS deassert commits the transaction.
//
// Ports
//   clk           -- input,  1b,    system clock
//   rst           -- input,  1b,    synchronous active-high reset
//   sck           -- input,  1b,    SPI clock (from host)
//   cs_n          -- input,  1b,    SPI chip-select, active-low
//   mosi          -- input,  1b,    master-out slave-in
//   miso          -- output, 1b,    master-in slave-out
//   act_out       -- output, Nb,    unpacked activation vector to compute core
//   wgt_out       -- output, Nb,    unpacked weight vector to compute core
//   compute_start -- output, 1b,    one-cycle pulse when CTRL reg written 0x01
//   result_out    -- input,  1b,    neuron result from compute core
//   result_valid  -- input,  1b,    result valid flag from compute core
//
// Register map (7-bit address, 128 entries):
//   0x00–(ACT_BYTES-1)             activation bytes (bit-packed, LSB first)
//   ACT_BYTES–(ACT_BYTES+WGT_BYTES-1) weight bytes
//   0x7E                           CTRL   — write 0x01 → pulse compute_start
//   0x7F                           STATUS — bit1=result_valid, bit0=result_out (read-only)
//
// Clock domain: single clock (clk). SPI pins synchronised via 2-FF stages.
// Reset: synchronous, active-high.
// Protocol conformance: SPI Mode 0 (CPOL=0, CPHA=0). Data sampled on rising SCK,
//   shifted out on falling SCK. Address auto-increments for burst transfers.

module interface_module #(
    parameter int N        = 64,
    parameter int ACT_BASE = 0,
    parameter int WGT_BASE = (N + 7) / 8
) (
    input  logic        clk,
    input  logic        rst,

    input  logic        sck,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    output logic [N-1:0] act_out,
    output logic [N-1:0] wgt_out,
    output logic         compute_start,
    input  logic         result_out,
    input  logic         result_valid
);

    localparam int ACT_BYTES = (N + 7) / 8;
    localparam int WGT_BYTES = (N + 7) / 8;

    // 2-FF synchronisers for SPI pins
    logic sck_s0, sck_s1, cs_s0, cs_s1, mosi_s0, mosi_s1;

    always_ff @(posedge clk) begin
        if (rst) begin
            {sck_s1, sck_s0}   <= 2'b00;
            {cs_s1,  cs_s0}    <= 2'b11;
            {mosi_s1, mosi_s0} <= 2'b00;
        end else begin
            sck_s0 <= sck;   sck_s1 <= sck_s0;
            cs_s0  <= cs_n;  cs_s1  <= cs_s0;
            mosi_s0 <= mosi; mosi_s1 <= mosi_s0;
        end
    end

    wire sck_rise  = ( sck_s0 & ~sck_s1);
    wire sck_fall  = (~sck_s0 &  sck_s1);
    wire cs_active = ~cs_s1;

    logic [7:0] regfile [0:127];

    logic [2:0]  bit_cnt;
    logic [6:0]  byte_idx;
    logic [7:0]  rx_byte;
    logic [7:0]  tx_byte;
    logic        wr_flag;
    logic [6:0]  cur_addr;
    logic        cs_active_prev;

    always_ff @(posedge clk) begin
        if (rst) begin
            bit_cnt        <= 3'd0;
            byte_idx       <= 7'd0;
            rx_byte        <= 8'b0;
            tx_byte        <= 8'b0;
            wr_flag        <= 1'b0;
            cur_addr       <= 7'd0;
            miso           <= 1'b0;
            compute_start  <= 1'b0;
            act_out        <= '0;
            wgt_out        <= '0;
            cs_active_prev <= 1'b0;
            for (int i = 0; i < 128; i++) regfile[i] <= 8'b0;
        end else begin
            compute_start  <= 1'b0;
            cs_active_prev <= cs_active;

            regfile[7'h7F] <= {6'b0, result_valid, result_out};

            if (cs_active_prev && !cs_active) begin
                bit_cnt  <= 3'd0;
                byte_idx <= 7'd0;
                rx_byte  <= 8'b0;
                miso     <= 1'b0;
            end

            if (cs_active) begin
                if (sck_rise) begin
                    rx_byte <= {rx_byte[6:0], mosi_s1};

                    if (bit_cnt == 3'd7) begin
                        if (byte_idx == 7'd0) begin
                            wr_flag  <= rx_byte[6];
                            cur_addr <= {rx_byte[5:0], mosi_s1};
                            tx_byte  <= regfile[{rx_byte[5:0], mosi_s1}];
                        end else begin
                            if (wr_flag) begin
                                regfile[cur_addr] <= {rx_byte[6:0], mosi_s1};

                                if (cur_addr >= 7'(ACT_BASE) &&
                                    cur_addr <  7'(ACT_BASE + ACT_BYTES))
                                    act_out[(cur_addr - 7'(ACT_BASE))*8 +: 8]
                                        <= {rx_byte[6:0], mosi_s1};

                                if (cur_addr >= 7'(WGT_BASE) &&
                                    cur_addr <  7'(WGT_BASE + WGT_BYTES))
                                    wgt_out[(cur_addr - 7'(WGT_BASE))*8 +: 8]
                                        <= {rx_byte[6:0], mosi_s1};

                                if (cur_addr == 7'h7E && mosi_s1)
                                    compute_start <= 1'b1;
                            end
                            tx_byte  <= regfile[cur_addr + 7'd1];
                            cur_addr <= cur_addr + 7'd1;
                        end

                        byte_idx <= byte_idx + 7'd1;
                        bit_cnt  <= 3'd0;
                        rx_byte  <= 8'b0;
                    end else begin
                        bit_cnt <= bit_cnt + 3'd1;
                    end
                end

                if (sck_fall && byte_idx >= 7'd1) begin
                    miso    <= tx_byte[7];
                    tx_byte <= {tx_byte[6:0], 1'b0};
                end
            end
        end
    end

endmodule
