"""Run SPI slave cocotb tests via cocotb_tools runner."""
from pathlib import Path
from cocotb_tools.runner import get_runner

hdl_dir = Path(__file__).parent

runner = get_runner("icarus")
runner.build(
    verilog_sources=[hdl_dir / "spi_slave.sv"],
    hdl_toplevel="spi_slave",
    build_args=["-g2012"],
    build_dir=hdl_dir / "sim_build_spi",
    always=True,
)
runner.test(
    hdl_toplevel="spi_slave",
    test_module="test_spi_slave",
    build_dir=hdl_dir / "sim_build_spi",
    results_xml=str(hdl_dir / "results_spi.xml"),
)
