import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

N = 64   # must match parameter in bnn_layer.sv


def py_bnn_layer(activation_bits: int, weight_bits: int, n: int) -> int:
    """Python golden model: returns 1 if popcount(XNOR) > N/2 else 0"""
    xnor = ~(activation_bits ^ weight_bits) & ((1 << n) - 1)
    pop = bin(xnor).count("1")
    return 1 if pop > n // 2 else 0


@cocotb.test()
async def test_bnn_reset(dut):
    """Drive reset, verify result_valid is low"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.act_valid.value = 0
    dut.weight_valid.value = 0
    dut.activation.value = 0
    dut.weight_row.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert dut.result_valid.value == 0, "result_valid should be 0 after reset"
    dut.rst.value = 0


@cocotb.test()
async def test_bnn_all_agree(dut):
    """All activations match weights — expect out=1 (all XNOR bits = 1, pop=N)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    pattern = (1 << N) - 1  # all ones
    dut.activation.value = pattern
    dut.weight_row.value = pattern
    dut.act_valid.value = 1
    dut.weight_valid.value = 1

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.result_valid.value == 1
    assert dut.out.value == 1, f"Expected 1 (all agree), got {dut.out.value}"


@cocotb.test()
async def test_bnn_random(dut):
    """Compare RTL against Python golden model on 5 random inputs"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    for i in range(5):
        act = random.getrandbits(N)
        wgt = random.getrandbits(N)
        expected = py_bnn_layer(act, wgt, N)

        dut.activation.value = act
        dut.weight_row.value = wgt
        dut.act_valid.value = 1
        dut.weight_valid.value = 1

        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

        got = int(dut.out.value)
        assert got == expected, f"Test {i}: act={act:#x} wgt={wgt:#x} expected={expected} got={got}"
        dut._log.info(f"Test {i} PASS: out={got} expected={expected}")
