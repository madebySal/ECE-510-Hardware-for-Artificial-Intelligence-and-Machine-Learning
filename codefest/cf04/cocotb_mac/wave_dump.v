// Companion module: enables VCD waveform dump for GTKWave
module wave_dump;
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, mac);
    end
endmodule
