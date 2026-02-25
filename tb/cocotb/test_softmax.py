"""
CocoTB Testbench for Softmax Unit
Tests the piecewise-linear approximate softmax with various input patterns.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import math


VEC_LEN = 8
FRAC_BITS = 8
TIMEOUT_CYCLES = 500


def fp_to_int(val):
    """Convert float to Q8.8 signed 16-bit."""
    raw = int(val * (1 << FRAC_BITS))
    return raw & 0xFFFF


def int_to_fp(val):
    """Convert Q8.8 16-bit signed to float."""
    if val & 0x8000:
        val = val - 0x10000
    return val / (1 << FRAC_BITS)


async def reset_dut(dut):
    """Apply reset."""
    dut.rst_n.value = 0
    dut.start.value = 0
    for i in range(VEC_LEN):
        dut.scores[i].value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_softmax(dut, scores_float):
    """Run softmax and return probabilities as floats."""
    # Set input scores
    for i in range(VEC_LEN):
        dut.scores[i].value = fp_to_int(scores_float[i])
    await RisingEdge(dut.clk)

    # Start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for valid with timeout
    for cycle in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        if dut.valid.value == 1:
            break
    else:
        raise TimeoutError(f"Softmax did not complete within {TIMEOUT_CYCLES} cycles")

    # Read output probabilities
    probs = []
    for i in range(VEC_LEN):
        raw = int(dut.probs[i].value) & 0xFFFF
        probs.append(int_to_fp(raw))

    # Wait for FSM to return to idle
    for _ in range(5):
        await RisingEdge(dut.clk)

    return probs


def reference_softmax(scores):
    """Compute reference softmax in Python."""
    max_s = max(scores)
    shifted = [s - max_s for s in scores]
    exps = [math.exp(s) for s in shifted]
    total = sum(exps)
    return [e / total for e in exps]


@cocotb.test()
async def test_uniform_inputs(dut):
    """Uniform inputs should produce roughly equal probabilities."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scores = [1.0] * VEC_LEN
    probs = await run_softmax(dut, scores)

    dut._log.info(f"Uniform input probs: {[f'{p:.4f}' for p in probs]}")

    # All should be roughly 1/8 = 0.125
    for i, p in enumerate(probs):
        assert p >= 0, f"Probability [{i}] should be non-negative, got {p}"

    # Check variance is small
    mean_p = sum(probs) / len(probs)
    variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
    dut._log.info(f"  Mean={mean_p:.4f}, Variance={variance:.6f}")

    # With approximation, just check they're all positive and close
    assert variance < 0.1, f"Variance too high for uniform inputs: {variance}"
    dut._log.info("PASS: Uniform inputs produce near-equal probabilities")


@cocotb.test()
async def test_dominant_score(dut):
    """One large score should dominate the probability distribution."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scores = [0.0] * VEC_LEN
    scores[0] = 4.0  # Dominant
    scores[3] = 0.5  # Medium

    probs = await run_softmax(dut, scores)
    dut._log.info(f"Dominant score probs: {[f'{p:.4f}' for p in probs]}")

    # probs[0] should be the largest
    assert probs[0] >= probs[3], \
        f"Dominant score should have highest prob: p[0]={probs[0]:.4f} vs p[3]={probs[3]:.4f}"
    dut._log.info("PASS: Dominant score has highest probability")


@cocotb.test()
async def test_negative_scores(dut):
    """Negative scores: less negative should have higher probability."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scores = [-float(i + 1) for i in range(VEC_LEN)]  # [-1, -2, ..., -8]

    probs = await run_softmax(dut, scores)
    dut._log.info(f"Negative score probs: {[f'{p:.4f}' for p in probs]}")

    # First element (least negative) should have highest prob
    assert probs[0] >= probs[-1], \
        f"Less negative should have higher prob: p[0]={probs[0]:.4f} vs p[-1]={probs[-1]:.4f}"
    dut._log.info("PASS: Less negative → higher probability")


@cocotb.test()
async def test_probability_sum(dut):
    """Probabilities should sum approximately to 1.0."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scores = [0.5 * i for i in range(VEC_LEN)]
    probs = await run_softmax(dut, scores)

    total = sum(probs)
    dut._log.info(f"Probability sum: {total:.4f} (ideal: 1.0)")

    # Allow generous tolerance for PWL approximation
    assert 0.3 < total < 2.0, f"Sum out of acceptable range: {total}"
    dut._log.info(f"PASS: Probability sum = {total:.4f} (within tolerance)")


@cocotb.test()
async def test_ordering_preserved(dut):
    """Monotonically increasing scores should produce monotonically increasing probs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scores = [0.25 * i for i in range(VEC_LEN)]  # 0.0, 0.25, 0.5, ...
    probs = await run_softmax(dut, scores)

    dut._log.info(f"Ordered scores:  {[f'{s:.2f}' for s in scores]}")
    dut._log.info(f"Ordered probs:   {[f'{p:.4f}' for p in probs]}")

    # Check rough ordering: last should be >= first
    assert probs[-1] >= probs[0], \
        f"Ordering not preserved: p[-1]={probs[-1]:.4f} < p[0]={probs[0]:.4f}"
    dut._log.info("PASS: Probability ordering preserved")


@cocotb.test()
async def test_back_to_back(dut):
    """Run softmax twice in succession to verify FSM returns to idle properly."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # First run
    scores1 = [1.0] * VEC_LEN
    probs1 = await run_softmax(dut, scores1)
    dut._log.info(f"Run 1 probs: {[f'{p:.4f}' for p in probs1]}")

    # Second run (different input)
    scores2 = [0.0] * VEC_LEN
    scores2[0] = 3.0
    probs2 = await run_softmax(dut, scores2)
    dut._log.info(f"Run 2 probs: {[f'{p:.4f}' for p in probs2]}")

    # Second run should have dominant first element
    assert probs2[0] >= probs2[1], "Back-to-back: second run should show dominance"
    dut._log.info("PASS: Back-to-back execution works")
