SIM           = icarus
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(PWD)/../hdl/mac_correct.v $(PWD)/wave_dump.v
TOPLEVEL      = mac
MODULE        = test_mac
COMPILE_ARGS  += -s wave_dump

include $(shell cocotb-config --makefiles)/Makefile.sim
