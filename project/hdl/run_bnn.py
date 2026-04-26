"""Run BNN layer cocotb tests via cocotb_tools runner."""
from pathlib import Path
from cocotb_tools.runner import get_runner

hdl_dir = Path(__file__).parent

runner = get_runner("icarus")
runner.build(
    verilog_sources=[hdl_dir / "bnn_layer.sv"],
    hdl_toplevel="bnn_layer",
    build_args=["-g2012"],
    build_dir=hdl_dir / "sim_build_bnn",
    always=True,
)
runner.test(
    hdl_toplevel="bnn_layer",
    test_module="test_bnn_layer",
    build_dir=hdl_dir / "sim_build_bnn",
    results_xml=str(hdl_dir / "results_bnn.xml"),
)
