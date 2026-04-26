import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def clk_cycle(dut):
    """Advance one clock cycle and settle past the NBA region."""
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")


async def reset(dut):
    dut.rst.value = 1
    dut.a.value = 0
    dut.b.value = 0
    await clk_cycle(dut)
    dut.rst.value = 0


@cocotb.test()
async def test_mac_basic(dut):
    """[a=3,b=4] x3 cycles, reset, [a=-5,b=2] x2 cycles"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.a.value = 3
    dut.b.value = 4
    for expected in [12, 24, 36]:
        await clk_cycle(dut)
        got = dut.out.value.to_signed()
        assert got == expected, f"Expected {expected}, got {got}"

    # assert reset
    dut.rst.value = 1
    await clk_cycle(dut)
    assert dut.out.value.to_signed() == 0, f"Reset failed: {dut.out.value.to_signed()}"
    dut.rst.value = 0

    # negative inputs: a=-5, b=2 → expect -10, -20
    dut.a.value = -5   # cocotb v2 accepts signed integers directly
    dut.b.value = 2
    for expected in [-10, -20]:
        await clk_cycle(dut)
        got = dut.out.value.to_signed()
        assert got == expected, f"Expected {expected}, got {got}"


@cocotb.test()
async def test_mac_overflow(dut):
    """Accumulate until 32-bit signed overflow; documents wrap-around (no saturation)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # a=127, b=127 → product=16129 per cycle
    # overflow after floor(2^31 / 16129) = 133144 cycles
    dut.a.value = 127
    dut.b.value = 127
    n_before = (1 << 31) // (127 * 127)  # 133144

    for _ in range(n_before):
        await RisingEdge(dut.clk)

    await Timer(1, unit="ps")
    val_before = dut.out.value.to_signed()

    await clk_cycle(dut)
    val_after = dut.out.value.to_signed()

    if val_after < 0 < val_before:
        dut._log.info(
            f"OVERFLOW WRAP (expected): {val_before} -> {val_after}. "
            "Design wraps (2's complement), does NOT saturate."
        )
    else:
        dut._log.info(f"Overflow check: {val_before} -> {val_after}")
