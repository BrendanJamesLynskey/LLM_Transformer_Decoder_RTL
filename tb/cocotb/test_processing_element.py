"""
CocoTB Testbench for Processing Element
Tests MAC operations, data forwarding, and accumulator clear.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
import random


def fp_to_int(val):
    """Convert Q8.8 float to 16-bit signed integer."""
    raw = int(val * 256)
    if raw < 0:
        raw = raw & 0xFFFF
    return raw


def int_to_fp(val):
    """Convert 16-bit signed integer (Q8.8) to float."""
    if val & 0x8000:
        val = val - 0x10000
    return val / 256.0


async def reset_dut(dut):
    """Apply reset sequence."""
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset(dut):
    """Test that reset clears the accumulator."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    acc = int(dut.acc_out.value)
    assert acc == 0, f"After reset, acc should be 0, got {acc}"
    dut._log.info("PASS: Reset clears accumulator")


@cocotb.test()
async def test_simple_mac(dut):
    """Test basic multiply-accumulate: 2.0 * 3.0 = 6.0."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Feed 2.0 * 3.0
    dut.enable.value = 1
    dut.a_in.value = fp_to_int(2.0)  # 0x0200
    dut.w_in.value = fp_to_int(3.0)  # 0x0300
    await RisingEdge(dut.clk)

    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    await RisingEdge(dut.clk)

    acc = int(dut.acc_out.value)
    # In full precision: 0x0200 * 0x0300 = 0x60000
    # 2.0 * 3.0 = 6.0 → 6.0 * 256 * 256 = 393216 = 0x60000
    expected = 0x0200 * 0x0300  # 393216
    dut._log.info(f"MAC result: {acc:#010x}, expected: {expected:#010x}")
    assert acc == expected, f"MAC failed: got {acc:#x}, expected {expected:#x}"
    dut._log.info("PASS: Simple MAC (2.0 * 3.0)")


@cocotb.test()
async def test_accumulation(dut):
    """Test accumulation over multiple cycles."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Accumulate: 1.0 * 1.0 four times = 4.0
    for _ in range(4):
        dut.enable.value = 1
        dut.a_in.value = fp_to_int(1.0)
        dut.w_in.value = fp_to_int(1.0)
        await RisingEdge(dut.clk)

    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    await RisingEdge(dut.clk)

    acc = int(dut.acc_out.value)
    expected = 4 * (0x0100 * 0x0100)  # 4 * 65536 = 262144
    dut._log.info(f"Accumulation: {acc}, expected {expected}")
    assert acc == expected, f"Accumulation failed: got {acc}, expected {expected}"
    dut._log.info("PASS: Accumulation (4 x 1.0*1.0 = 4.0)")


@cocotb.test()
async def test_clear(dut):
    """Test that clear resets the accumulator mid-operation."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Do some work
    dut.enable.value = 1
    dut.a_in.value = fp_to_int(5.0)
    dut.w_in.value = fp_to_int(5.0)
    await RisingEdge(dut.clk)
    dut.enable.value = 0
    await RisingEdge(dut.clk)

    # Verify non-zero
    acc_before = int(dut.acc_out.value)
    assert acc_before != 0, "Expected non-zero accumulator before clear"

    # Clear
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    acc_after = int(dut.acc_out.value)
    assert acc_after == 0, f"After clear, acc should be 0, got {acc_after}"
    dut._log.info("PASS: Clear resets accumulator")


@cocotb.test()
async def test_data_forwarding(dut):
    """Test that a_in and w_in are forwarded to a_out and w_out."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    test_a = 0x00AB
    test_w = 0x00CD

    dut.enable.value = 1
    dut.a_in.value = test_a
    dut.w_in.value = test_w
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    a_out = int(dut.a_out.value) & 0xFFFF
    w_out = int(dut.w_out.value) & 0xFFFF

    assert a_out == test_a, f"a_out forwarding: got {a_out:#x}, expected {test_a:#x}"
    assert w_out == test_w, f"w_out forwarding: got {w_out:#x}, expected {test_w:#x}"
    dut._log.info("PASS: Data forwarding correct")


@cocotb.test()
async def test_negative_numbers(dut):
    """Test MAC with negative fixed-point numbers."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # -2.0 * 3.0 = -6.0
    dut.enable.value = 1
    dut.a_in.value = fp_to_int(-2.0)  # 0xFE00
    dut.w_in.value = fp_to_int(3.0)   # 0x0300
    await RisingEdge(dut.clk)

    dut.enable.value = 0
    await RisingEdge(dut.clk)

    acc_raw = int(dut.acc_out.value)
    # Sign-extend 32-bit
    if acc_raw & 0x80000000:
        acc_signed = acc_raw - 0x100000000
    else:
        acc_signed = acc_raw

    # -2.0 * 3.0 in Q8.8 full precision = -0x60000 = -393216
    dut._log.info(f"Negative MAC: acc = {acc_signed} ({acc_raw:#010x})")
    assert acc_signed < 0, "Expected negative result for -2.0 * 3.0"
    dut._log.info("PASS: Negative number MAC")


@cocotb.test()
async def test_random_mac(dut):
    """Randomized MAC test with golden model comparison."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    golden_acc = 0
    num_ops = 10

    for i in range(num_ops):
        a_float = random.uniform(-4.0, 4.0)
        w_float = random.uniform(-4.0, 4.0)

        a_q88 = max(-128, min(127, int(a_float * 256)))
        w_q88 = max(-128, min(127, int(w_float * 256)))

        a_16 = a_q88 & 0xFFFF
        w_16 = w_q88 & 0xFFFF

        # Sign-extend for golden model
        a_signed = a_q88 if a_q88 >= 0 else a_q88
        w_signed = w_q88 if w_q88 >= 0 else w_q88
        golden_acc += a_signed * w_signed

        dut.enable.value = 1
        dut.a_in.value = a_16
        dut.w_in.value = w_16
        await RisingEdge(dut.clk)

    dut.enable.value = 0
    await RisingEdge(dut.clk)

    hw_acc = int(dut.acc_out.value)
    if hw_acc & 0x80000000:
        hw_acc_signed = hw_acc - 0x100000000
    else:
        hw_acc_signed = hw_acc

    golden_32 = golden_acc & 0xFFFFFFFF
    if golden_32 & 0x80000000:
        golden_signed = golden_32 - 0x100000000
    else:
        golden_signed = golden_32

    dut._log.info(f"Random MAC ({num_ops} ops): HW={hw_acc_signed}, Golden={golden_signed}")
    # Allow small tolerance for potential rounding
    assert abs(hw_acc_signed - golden_signed) < 16, \
        f"Random MAC mismatch: HW={hw_acc_signed}, Golden={golden_signed}"
    dut._log.info("PASS: Random MAC matches golden model")
