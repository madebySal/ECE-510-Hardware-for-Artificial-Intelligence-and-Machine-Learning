"""End-to-end cocotb testbench for bnn_top.sv.

Burst SPI protocol:
  - CS asserted, CMD byte [W/R | ADDR[6:0]], then N data bytes (auto-increment)
  - CS deasserted to commit
  - For reads: CMD byte + N dummy bytes; data clocked out on MISO

Register map (N=64 default → ACT_BYTES=WGT_BYTES=8):
  0x00–0x07  activation bytes (byte 0 = bits[7:0])
  0x08–0x0F  weight bytes     (byte 0 = bits[7:0])
  0x7E       CTRL             (write 0x01 → compute_start)
  0x7F       STATUS           (bit1=result_valid, bit0=result_out)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

N         = 64
ACT_BYTES = (N + 7) // 8   # 8
WGT_BYTES = (N + 7) // 8   # 8
ACT_BASE  = 0x00
WGT_BASE  = ACT_BYTES      # 0x08
SCK_HALF  = 40              # ns


def py_bnn(act_int: int, wgt_int: int) -> int:
    mask = (1 << N) - 1
    xnor = ~(act_int ^ wgt_int) & mask
    return 1 if bin(xnor).count("1") > N // 2 else 0


async def spi_burst(dut, wr: bool, addr: int, data_bytes: list) -> list:
    """Burst SPI transaction.  Returns list of received bytes (reads only)."""
    cmd = (0x80 | (addr & 0x7F)) if wr else (addr & 0x7F)
    frame = [cmd] + list(data_bytes)

    dut.cs_n.value = 0
    await Timer(SCK_HALF, unit="ns")

    rx_bytes = []
    for byte_val in frame:
        rx = 0
        for bit in range(7, -1, -1):
            dut.mosi.value = (byte_val >> bit) & 1
            await Timer(SCK_HALF, unit="ns")
            dut.sck.value = 1
            await Timer(SCK_HALF, unit="ns")
            rx = (rx << 1) | int(dut.miso.value)
            dut.sck.value = 0
            await Timer(SCK_HALF, unit="ns")
        rx_bytes.append(rx & 0xFF)

    dut.cs_n.value = 1
    dut.mosi.value = 0
    await Timer(SCK_HALF * 4, unit="ns")
    return rx_bytes[1:]   # drop the cmd-phase echo bytes


async def spi_reset(dut):
    dut.rst.value  = 1
    dut.sck.value  = 0
    dut.cs_n.value = 1
    dut.mosi.value = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def run_inference(dut, act_int: int, wgt_int: int) -> int:
    """Load activation + weight via burst SPI, trigger, poll result."""
    act_bytes = [(act_int >> (i * 8)) & 0xFF for i in range(ACT_BYTES)]
    wgt_bytes = [(wgt_int >> (i * 8)) & 0xFF for i in range(WGT_BYTES)]

    # Burst-write activation (one SPI transaction)
    await spi_burst(dut, wr=True, addr=ACT_BASE, data_bytes=act_bytes)

    # Burst-write weight row (one SPI transaction)
    await spi_burst(dut, wr=True, addr=WGT_BASE, data_bytes=wgt_bytes)

    # Write CTRL = 0x01 (trigger compute)
    await spi_burst(dut, wr=True, addr=0x7E, data_bytes=[0x01])

    # Poll STATUS until result_valid (bit1) set
    for _ in range(20):
        rx = await spi_burst(dut, wr=False, addr=0x7F, data_bytes=[0x00])
        status = rx[0]
        if status & 0x02:
            return status & 0x01
    raise AssertionError("Timeout waiting for result_valid")


@cocotb.test()
async def test_all_agree(dut):
    """All activations match weights → pop=N → out=1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await spi_reset(dut)

    act = (1 << N) - 1
    wgt = (1 << N) - 1
    result   = await run_inference(dut, act, wgt)
    expected = py_bnn(act, wgt)
    assert result == expected == 1, f"Expected 1, got {result}"
    dut._log.info("PASS all_agree: out=1")


@cocotb.test()
async def test_all_disagree(dut):
    """All activations oppose weights → pop=0 → out=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await spi_reset(dut)

    act = (1 << N) - 1   # all 1s
    wgt = 0               # all 0s → XNOR=0 everywhere → pop=0
    result   = await run_inference(dut, act, wgt)
    expected = py_bnn(act, wgt)
    assert result == expected == 0, f"Expected 0, got {result}"
    dut._log.info("PASS all_disagree: out=0")


@cocotb.test()
async def test_half_agree(dut):
    """Exactly N/2 agreements → pop = N/2 → threshold not met → out=0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await spi_reset(dut)

    # First N/2 bits agree, last N/2 bits disagree
    agree_mask = (1 << (N // 2)) - 1
    act = agree_mask
    wgt = agree_mask
    result   = await run_inference(dut, act, wgt)
    expected = py_bnn(act, wgt)
    assert result == expected, f"Expected {expected}, got {result}"
    dut._log.info(f"PASS half_agree: out={result} (threshold not met)")


@cocotb.test()
async def test_random_inputs(dut):
    """5 random (act, wgt) pairs compared against Python golden model."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await spi_reset(dut)

    mask = (1 << N) - 1
    for i in range(5):
        act = random.getrandbits(N)
        wgt = random.getrandbits(N)
        result   = await run_inference(dut, act, wgt)
        expected = py_bnn(act, wgt)
        assert result == expected, \
            f"Test {i}: act={act:#018x} wgt={wgt:#018x} expected={expected} got={result}"
        dut._log.info(f"PASS random {i}: out={result} expected={expected}")
