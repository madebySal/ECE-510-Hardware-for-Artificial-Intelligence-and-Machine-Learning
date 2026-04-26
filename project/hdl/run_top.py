"""Run end-to-end bnn_top cocotb tests."""
from pathlib import Path
from cocotb_tools.runner import get_runner

hdl = Path(__file__).parent

runner = get_runner("icarus")
runner.build(
    sources=[hdl / "bnn_layer.sv", hdl / "spi_slave.sv", hdl / "bnn_top.sv"],
    hdl_toplevel="bnn_top",
    build_args=["-g2012"],
    build_dir=hdl / "sim_build_top",
    always=True,
)
runner.test(
    hdl_toplevel="bnn_top",
    test_module="test_bnn_top",
    build_dir=hdl / "sim_build_top",
    results_xml=str(hdl / "results_top.xml"),
)
