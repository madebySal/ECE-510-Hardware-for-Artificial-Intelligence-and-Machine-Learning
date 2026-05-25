# SDC constraints for BNN top module — sky130A @ 100 MHz
create_clock -name clk -period 10.0 [get_ports clk]

set_input_delay  -clock clk 2.0 [get_ports {sck cs_n mosi rst}]
set_output_delay -clock clk 2.0 [get_ports miso]

# SPI pins are synchronised internally via 2-FF stages;
# they are treated as multi-cycle paths from the clock perspective.
set_multicycle_path -setup 2 -from [get_ports {sck cs_n mosi}]
set_multicycle_path -hold  1 -from [get_ports {sck cs_n mosi}]

set_false_path -from [get_ports rst]
