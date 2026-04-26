"""cocotb testbench for spi_slave.sv — burst protocol (N=64 default)."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

N         = 64
ACT_BYTES = (N + 7) // 8   # 8
WGT_BYTES = (N + 7) // 8   # 8
SCK_HALF  = 40              # ns — 12.5 MHz SCK


async def spi_burst(dut, wr: bool, addr: int, data_bytes: list) -> list:
    """Burst SPI frame: CMD byte + data_bytes.  Returns received bytes."""
    cmd   = (0x80 | (addr & 0x7F)) if wr else (addr & 0x7F)
    frame = [cmd] + list(data_bytes)

    dut.cs_n.value = 0
    await Timer(SCK_HALF, unit="ns")

    rx_bytes = []
    for bval in frame:
        rx = 0
        for bit in range(7, -1, -1):
            dut.mosi.value = (bval >> bit) & 1
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
    return rx_bytes[1:]


async def hw_reset(dut):
    dut.rst.value          = 1
    dut.sck.value          = 0
    dut.cs_n.value         = 1
    dut.mosi.value         = 0
    dut.result_out.value   = 0
    dut.result_valid.value = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_burst_write_read(dut):
    """Burst-write 4 bytes starting at addr 0x00, read them back."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await hw_reset(dut)

    payload = [0xAB, 0xCD, 0xEF, 0x12]

    # Write burst: CMD=0x80 (write, addr=0), then 4 bytes
    await spi_burst(dut, wr=True, addr=0x00, data_bytes=payload)

    # Read burst: CMD=0x00 (read, addr=0), 4 dummy bytes → get stored values
    rx = await spi_burst(dut, wr=False, addr=0x00, data_bytes=[0]*4)

    assert rx == payload, f"Read back {[hex(b) for b in rx]}, expected {[hex(b) for b in payload]}"
    dut._log.info(f"PASS burst write/read: {[hex(b) for b in rx]}")


@cocotb.test()
async def test_act_wgt_unpack(dut):
    """Write 8 activation bytes; verify act_out assembles correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await hw_reset(dut)

    act_val = 0xDEAD_BEEF_CAFE_1234
    act_bytes = [(act_val >> (i * 8)) & 0xFF for i in range(ACT_BYTES)]

    await spi_burst(dut, wr=True, addr=0x00, data_bytes=act_bytes)

    # Allow register writes to propagate
    for _ in range(4):
        await RisingEdge(dut.clk)

    got = int(dut.act_out.value)
    assert got == act_val, f"act_out=0x{got:016X}, expected=0x{act_val:016X}"
    dut._log.info(f"PASS act_wgt_unpack: act_out=0x{got:016X}")


@cocotb.test()
async def test_result_read(dut):
    """Drive result_in=0x01, result_valid=1; verify STATUS reg."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await hw_reset(dut)

    dut.result_out.value   = 0x01
    dut.result_valid.value = 1
    for _ in range(4):
        await RisingEdge(dut.clk)

    rx = await spi_burst(dut, wr=False, addr=0x7F, data_bytes=[0x00])
    status = rx[0]
    assert status & 0x02, f"result_valid bit not set: STATUS=0x{status:02X}"
    assert status & 0x01, f"result_out bit not set:   STATUS=0x{status:02X}"
    dut._log.info(f"PASS result_read: STATUS=0x{status:02X}")
