`timescale 1ns/1ps
// SPI Mode-0 slave — burst-capable register file for BNN inference control
//
// Protocol (MSB first, CS active-low):
//   Byte 0:    [W/R | ADDR[6:0]]   — command byte; sets base address
//   Bytes 1..: data bytes (write→stored; read→clocked out on MISO)
//              address auto-increments after each byte (burst transfer)
//   CS deassert commits the transaction.
//
// Register map (7-bit, 128 entries):
//   0x00–0x61  (0–97)   ACT[7:0]     activation bytes, byte-0=bits[7:0]
//   0x62–0x63  (98–99)  RESERVED     (weight loading reserved for extended map)
//   0x7C                WADDR        weight burst base (default 0; see note)
//   0x7D                reserved
//   0x7E                CTRL         write 0x01 → pulse compute_start
//   0x7F                STATUS       bit1=result_valid, bit0=result_out (read-only)
//
// Weight loading: write to addresses 0x00-0x61 with W/R=1 and addr MSB=1
//   (ADDR[6]=1 selects weight bank; ADDR[5:0] = byte offset 0-97)
//   Activation: ADDR[6]=0, Weight: ADDR[6]=1 → two 64-byte banks, total 98 each
//   Note: activation bank uses 0x00-0x61 (byte offsets 0-96, 97 bytes + 1 extra)
//         weight bank uses 0x40-0x61 + wrap (see note below)
//
// Simplified flat map actually used in this implementation:
//   CMD byte ADDR[6:0]:
//     0x00-0x60  Activation bytes 0-96  (97 bytes → bits 775..0)
//     0x61       Activation byte  97    (bits 783..776)
//     0x62-0x7B  Weight bytes     0-25  (partial — for narrow-N demos)
//   For full N=784 weights in simulation: set N=64 (default) or use
//   WADDR-based burst (addr 0x7C sets weight burst destination).
//
// *** Implementation uses N as parameter; default N=64 fits in 7-bit addr space.
//     For N=784 in silicon, extend to 8-bit address (3-byte frame). ***

module spi_slave #(
    parameter int N        = 64,
    parameter int ACT_BASE = 0,
    parameter int WGT_BASE = (N + 7) / 8    // immediately after activation bytes
) (
    input  logic        clk,
    input  logic        rst,

    // SPI pins
    input  logic        sck,
    input  logic        cs_n,
    input  logic        mosi,
    output logic        miso,

    // Compute-core interface
    output logic [N-1:0] act_out,
    output logic [N-1:0] wgt_out,
    output logic         compute_start,
    input  logic         result_out,
    input  logic         result_valid
);

    localparam int ACT_BYTES = (N + 7) / 8;
    localparam int WGT_BYTES = (N + 7) / 8;

    // -----------------------------------------------------------------------
    // 2-FF synchronisers
    // -----------------------------------------------------------------------
    logic sck_s0, sck_s1, cs_s0, cs_s1, mosi_s0, mosi_s1;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            {sck_s1, sck_s0}   <= 2'b00;
            {cs_s1,  cs_s0}    <= 2'b11;
            {mosi_s1, mosi_s0} <= 2'b00;
        end else begin
            sck_s0 <= sck;  sck_s1 <= sck_s0;
            cs_s0  <= cs_n; cs_s1  <= cs_s0;
            mosi_s0 <= mosi; mosi_s1 <= mosi_s0;
        end
    end

    wire sck_rise  = ( sck_s0 & ~sck_s1);
    wire sck_fall  = (~sck_s0 &  sck_s1);
    wire cs_active = ~cs_s1;

    // -----------------------------------------------------------------------
    // 128-byte register file
    // -----------------------------------------------------------------------
    logic [7:0] regfile [0:127];

    // -----------------------------------------------------------------------
    // Burst shift state
    // -----------------------------------------------------------------------
    logic [2:0]  bit_cnt;      // position within current byte (0..7)
    logic [6:0]  byte_idx;     // which byte within frame (0=cmd, 1..=data)
    logic [7:0]  rx_byte;      // accumulates current byte
    logic [7:0]  tx_byte;      // current byte being shifted out
    logic        wr_flag;      // latched from cmd byte
    logic [6:0]  base_addr;    // latched from cmd byte
    logic [6:0]  cur_addr;     // auto-increments per data byte

    logic        cs_active_prev;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            bit_cnt        <= 3'd0;
            byte_idx       <= 7'd0;
            rx_byte        <= 8'b0;
            tx_byte        <= 8'b0;
            wr_flag        <= 1'b0;
            base_addr      <= 7'd0;
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

            // Mirror read-only STATUS
            regfile[7'h7F] <= {6'b0, result_valid, result_out};

            if (cs_active_prev && !cs_active) begin
                // CS deasserted — reset for next frame
                bit_cnt  <= 3'd0;
                byte_idx <= 7'd0;
                rx_byte  <= 8'b0;
                miso     <= 1'b0;
            end

            if (cs_active) begin
                if (sck_rise) begin
                    rx_byte <= {rx_byte[6:0], mosi_s1};

                    if (bit_cnt == 3'd7) begin
                        // Completed a byte — full_byte = {rx_byte[6:0], mosi_s1}
                        if (byte_idx == 7'd0) begin
                            // CMD byte: bit7=W/R, bits[6:0]=ADDR
                            wr_flag   <= rx_byte[6];           // MSB arrived first → sits at rx_byte[6]
                            base_addr <= {rx_byte[5:0], mosi_s1};
                            cur_addr  <= {rx_byte[5:0], mosi_s1};
                            tx_byte   <= regfile[{rx_byte[5:0], mosi_s1}];
                        end else begin
                            // DATA byte
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

                // Shift MISO out on falling edge (skip cmd byte)
                if (sck_fall && byte_idx >= 7'd1) begin
                    miso    <= tx_byte[7];
                    tx_byte <= {tx_byte[6:0], 1'b0};
                end
            end
        end
    end

endmodule
