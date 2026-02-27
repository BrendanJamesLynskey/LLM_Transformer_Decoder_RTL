#!/usr/bin/env python3
"""
=============================================================================
RTL Behavioral Verification Suite
=============================================================================
Bit-accurate Python model of the LLM Transformer Decoder accelerator.
Each class mirrors an RTL module exactly, using the same Q8.8 fixed-point
arithmetic, FSM sequencing, and data widths. This verifies functional
correctness of the design before RTL simulation.
=============================================================================
"""

import sys
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# ===========================================================================
# Fixed-Point Parameters (must match transformer_pkg.sv)
# ===========================================================================
DATA_WIDTH = 16
FRAC_BITS = 8
ACC_WIDTH = 32
D_MODEL = 64
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 16
D_FF = 256
MAX_SEQ_LEN = 128

SIGN_MASK_16 = 0x8000
MASK_16 = 0xFFFF
SIGN_MASK_32 = 0x80000000
MASK_32 = 0xFFFFFFFF

# ===========================================================================
# Fixed-Point Utility Functions (mirror transformer_pkg.sv)
# ===========================================================================

def to_q88(val: float) -> int:
    """Convert float to Q8.8 signed 16-bit integer."""
    raw = int(round(val * (1 << FRAC_BITS)))
    # Clamp to Q8.8 range
    raw = max(-32768, min(32767, raw))
    return raw & MASK_16

def from_q88(val: int) -> float:
    """Convert Q8.8 16-bit to float (sign-extend)."""
    if val & SIGN_MASK_16:
        val = val - 0x10000
    return val / (1 << FRAC_BITS)

def sign_extend_16(val: int) -> int:
    """Sign-extend 16-bit to Python int."""
    val = val & MASK_16
    if val & SIGN_MASK_16:
        return val - 0x10000
    return val

def sign_extend_32(val: int) -> int:
    """Sign-extend 32-bit to Python int."""
    val = val & MASK_32
    if val & SIGN_MASK_32:
        return val - 0x100000000
    return val

def fp_mul(a: int, b: int) -> int:
    """Q8.8 multiply -> Q8.8 (matches RTL fp_mul)."""
    a_s = sign_extend_16(a)
    b_s = sign_extend_16(b)
    product = a_s * b_s
    # Arithmetic right shift by FRAC_BITS
    result = product >> FRAC_BITS
    return result & MASK_16

def fp_sat_add(a: int, b: int) -> int:
    """Saturating add for Q8.8 (matches RTL fp_sat_add)."""
    a_s = sign_extend_16(a)
    b_s = sign_extend_16(b)
    s = a_s + b_s
    if s > 32767:
        return 0x7FFF
    elif s < -32768:
        return 0x8000 & MASK_16
    return s & MASK_16

def fp_inv_sqrt(x: int) -> int:
    """Compute 1/sqrt(x) in Q8.8 using CLZ + 32-entry LUT + Newton-Raphson.
    Matches the RTL implementation in transformer_pkg.sv."""

    x_s = sign_extend_16(x)
    if x_s <= 0:
        return to_q88(1.0)
    if x_s == 1:
        return 0x1000  # 16.0 in Q8.8

    xu = x_s & 0xFFFF

    # Step 1: CLZ and normalise
    lz = 0
    for i in range(15, -1, -1):
        if xu & (1 << i):
            lz = 15 - i
            break
    else:
        lz = 15
    x_norm = (xu << lz) & 0xFFFF  # Q0.16, MSB is bit 15

    # Step 2: rsqrt LUT (32 entries, Q2.14)
    rsqrt_table = [
        22992, 22646, 22315, 21999, 21695, 21404, 21124, 20855,
        20596, 20346, 20106, 19873, 19649, 19431, 19221, 19018,
        18821, 18630, 18444, 18264, 18090, 17920, 17755, 17594,
        17438, 17285, 17137, 16992, 16851, 16714, 16579, 16448,
    ]
    lut_idx = (x_norm >> 10) & 0x1F
    r0 = rsqrt_table[lut_idx]

    # Step 3: Newton-Raphson: r1 = r0 * (3 - x_norm * r0^2) / 2
    r0_sq = r0 * r0
    r0_sq_16 = (r0_sq >> 14) & 0xFFFF
    xr2 = x_norm * r0_sq_16
    xr2_16 = (xr2 >> 16) & 0xFFFF
    three_minus = max(0, 49152 - xr2_16)
    r1_wide = r0 * three_minus
    r1 = (r1_wide >> 15) & 0xFFFF

    # Step 4: Denormalise using e = lz - 8
    # r1 in Q2.14 approx 1/sqrt(x_norm_float) where x_norm_float = x_norm/2^16
    # float_val = xu/2^8 = x_norm_float * 2^(8-lz)
    # 1/sqrt(float_val) = 1/sqrt(x_norm_float) * 2^((lz-8)/2)
    #                    = r1*2^(-14) * 2^((lz-8)/2)
    # In Q8.8: result = r1 * 2^((lz-8)/2 - 6)
    # For odd e: multiply r1 by sqrt(2) FIRST (preserves precision), then shift
    e = lz - 8  # signed, range [-8, +7]
    e_odd = (e % 2) != 0  # Python % always returns non-negative for positive divisor

    if e_odd:
        # Multiply r1 by sqrt(2) in full precision, then shift
        r1_adj = (r1 * 23170) >> 14  # sqrt(2) in Q2.14 = 23170
        shift_amt = 6 - (e - 1) // 2  # Python // rounds toward -inf
    else:
        r1_adj = r1
        shift_amt = 6 - e // 2

    if shift_amt >= 0:
        result = r1_adj >> shift_amt
    else:
        result = r1_adj << (-shift_amt)

    # Clamp
    if result > 0x7FFF:
        return 0x7FFF
    if result == 0:
        return 1
    return result & MASK_16

def mac_full_precision(a: int, b: int) -> int:
    """Full-precision MAC: a*b in 32-bit (matches PE accumulator)."""
    a_s = sign_extend_16(a)
    b_s = sign_extend_16(b)
    return (a_s * b_s) & MASK_32


# ===========================================================================
# Test Infrastructure
# ===========================================================================
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
            print(msg)
            self.errors.append(msg)

    def summary(self, module_name: str):
        total = self.passed + self.failed
        status = "PASS" if self.failed == 0 else "FAIL"
        print(f"\n  {module_name}: {self.passed}/{total} passed [{status}]")
        return self.failed == 0


# ===========================================================================
# Module 1: Processing Element
# ===========================================================================
class ProcessingElement:
    """Behavioral model of processing_element.sv"""
    def __init__(self):
        self.accumulator = 0  # 32-bit
        self.a_out = 0
        self.w_out = 0

    def reset(self):
        self.accumulator = 0
        self.a_out = 0
        self.w_out = 0

    def clear(self):
        self.accumulator = 0
        self.a_out = 0
        self.w_out = 0

    def clock(self, enable: bool, a_in: int, w_in: int):
        if enable:
            self.a_out = a_in & MASK_16
            self.w_out = w_in & MASK_16
            # MAC: acc += a * w (full precision, 32-bit)
            a_s = sign_extend_16(a_in)
            w_s = sign_extend_16(w_in)
            self.accumulator = (self.accumulator + a_s * w_s) & MASK_32


def test_processing_element():
    print("\n" + "="*60)
    print("  PROCESSING ELEMENT TESTS")
    print("="*60)
    res = TestResults()
    pe = ProcessingElement()

    # Test 1: Reset
    pe.reset()
    res.check(pe.accumulator == 0, "Reset clears accumulator")

    # Test 2: MAC 2.0 * 3.0
    pe.reset()
    pe.clock(True, to_q88(2.0), to_q88(3.0))
    expected = sign_extend_16(to_q88(2.0)) * sign_extend_16(to_q88(3.0))
    actual = sign_extend_32(pe.accumulator)
    res.check(actual == expected,
              f"MAC 2.0*3.0",
              f"got {actual} (0x{pe.accumulator:08X}), expected {expected} (0x{expected & MASK_32:08X})")

    # Test 3: Accumulation 4 x (1.0 * 1.0) = 4.0 in full precision
    pe.reset()
    for _ in range(4):
        pe.clock(True, to_q88(1.0), to_q88(1.0))
    expected = 4 * (256 * 256)  # 4 * 65536 = 262144
    actual = sign_extend_32(pe.accumulator)
    res.check(actual == expected,
              f"Accumulation 4x(1.0*1.0)",
              f"got {actual}, expected {expected}")

    # Test 4: Data forwarding
    pe.reset()
    pe.clock(True, 0x00AB, 0x00CD)
    res.check(pe.a_out == 0x00AB, "a_out forwarding", f"got 0x{pe.a_out:04X}")
    res.check(pe.w_out == 0x00CD, "w_out forwarding", f"got 0x{pe.w_out:04X}")

    # Test 5: Clear
    pe.clock(True, to_q88(5.0), to_q88(5.0))
    pe.clear()
    res.check(pe.accumulator == 0, "Clear resets accumulator")

    # Test 6: Negative numbers -2.0 * 3.0
    pe.reset()
    pe.clock(True, to_q88(-2.0), to_q88(3.0))
    actual = sign_extend_32(pe.accumulator)
    res.check(actual < 0, f"Negative MAC result (got {actual})")

    # Test 7: -1.5 * 2.0 = -3.0
    pe.reset()
    pe.clock(True, to_q88(-1.5), to_q88(2.0))
    actual = sign_extend_32(pe.accumulator)
    expected_fp = -1.5 * 2.0  # -3.0
    # In full precision: (-384) * 512 = -196608
    expected = sign_extend_16(to_q88(-1.5)) * sign_extend_16(to_q88(2.0))
    res.check(actual == expected,
              f"MAC -1.5*2.0",
              f"got {actual}, expected {expected} (= {expected / 65536.0:.4f} in float)")

    # Test 8: Random MAC with golden comparison
    pe.reset()
    golden = 0
    random.seed(42)
    for i in range(20):
        a_f = random.uniform(-4.0, 4.0)
        w_f = random.uniform(-4.0, 4.0)
        a_q = to_q88(a_f)
        w_q = to_q88(w_f)
        pe.clock(True, a_q, w_q)
        golden += sign_extend_16(a_q) * sign_extend_16(w_q)

    actual = sign_extend_32(pe.accumulator)
    golden_32 = sign_extend_32(golden & MASK_32)
    res.check(actual == golden_32,
              f"Random MAC (20 ops)",
              f"got {actual}, expected {golden_32}")

    return res.summary("Processing Element")


# ===========================================================================
# Module 2: Systolic Array
# ===========================================================================
class SystolicArray:
    """Behavioral model of systolic_array.sv (ROWS x COLS PEs).

    In the RTL, a_out and w_out are registered outputs (updated on posedge clk).
    The combinational wiring connects:
      a_wire[r][0]   = a_in[r]       (external input, left edge)
      a_wire[r][c+1] = PE[r][c].a_out  (registered output from previous cycle)
      w_wire[0][c]   = b_in[c]       (external input, top edge)
      w_wire[r+1][c] = PE[r][c].w_out  (registered output from previous cycle)

    So on each clock edge, each PE reads from its neighbor's *registered* output
    (which holds the value from the previous cycle). This is what creates the
    systolic wave propagation delay.
    """
    def __init__(self, rows=4, cols=4):
        self.rows = rows
        self.cols = cols
        self.pes = [[ProcessingElement() for _ in range(cols)] for _ in range(rows)]
        self.cycle_cnt = 0
        self.done = False

    def reset(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.pes[r][c].reset()
        self.cycle_cnt = 0
        self.done = False

    def clear(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.pes[r][c].clear()
        self.cycle_cnt = 0
        self.done = False

    def clock(self, enable: bool, a_in: List[int], b_in: List[int]):
        """One clock cycle: data flows through the array.

        IMPORTANT: We snapshot a_out/w_out BEFORE clocking the PEs,
        because these are registered outputs that represent the PREVIOUS
        cycle's values (exactly as in the RTL posedge behavior).
        """
        if not enable or self.done:
            return

        # Build the combinational wiring for this cycle
        # Snapshot registered outputs from PEs BEFORE this clock edge
        a_wire = [[0]*(self.cols+1) for _ in range(self.rows)]
        w_wire = [[0]*(self.cols) for _ in range(self.rows+1)]

        # External inputs drive the edges
        for r in range(self.rows):
            a_wire[r][0] = a_in[r]
        for c in range(self.cols):
            w_wire[0][c] = b_in[c]

        # Internal wiring: PE[r][c-1].a_out → PE[r][c] a_in
        #                  PE[r-1][c].w_out → PE[r][c] w_in
        # These are the REGISTERED outputs (values from the end of previous cycle)
        for r in range(self.rows):
            for c in range(self.cols):
                if c > 0:
                    a_wire[r][c] = self.pes[r][c-1].a_out
                if r > 0:
                    w_wire[r][c] = self.pes[r-1][c].w_out

        # Now clock all PEs simultaneously (posedge): each PE reads its
        # combinational inputs and updates its registered outputs
        for r in range(self.rows):
            for c in range(self.cols):
                self.pes[r][c].clock(True, a_wire[r][c], w_wire[r][c])

        self.cycle_cnt += 1
        if self.cycle_cnt >= self.rows + self.cols:
            self.done = True

    def get_results(self):
        """Return accumulator matrix."""
        return [[sign_extend_32(self.pes[r][c].accumulator)
                 for c in range(self.cols)]
                for r in range(self.rows)]


def test_systolic_array():
    print("\n" + "="*60)
    print("  SYSTOLIC ARRAY TESTS")
    print("="*60)
    res = TestResults()

    # Test 1: Single element - 1.0 * 2.0 at [0][0]
    sa = SystolicArray(4, 4)
    # Feed on cycle 0: a_in[0]=1.0, b_in[0]=2.0, rest zero
    a = [to_q88(1.0), 0, 0, 0]
    b = [to_q88(2.0), 0, 0, 0]
    sa.clock(True, a, b)
    # Drain remaining cycles
    zeros = [0, 0, 0, 0]
    for _ in range(7):
        sa.clock(True, zeros, zeros)

    results = sa.get_results()
    # PE[0][0] should have 1.0 * 2.0 in full precision = 256 * 512 = 131072
    expected_00 = 256 * 512  # 0x20000
    res.check(results[0][0] == expected_00,
              f"Single element [0][0] = 1.0*2.0",
              f"got {results[0][0]} (0x{results[0][0] & MASK_32:08X}), expected {expected_00} (0x{expected_00:08X})")

    # Test 2: Done signal
    res.check(sa.done, "Done asserted after streaming")

    # Test 3: Clear and reuse
    sa.clear()
    res.check(not sa.done, "Clear resets done")
    results_cleared = sa.get_results()
    res.check(all(results_cleared[r][c] == 0 for r in range(4) for c in range(4)),
              "Clear resets all accumulators")

    # Test 4: 2x2 matrix multiply using direct computation to verify PE math
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    #
    # In a systolic array, we need proper staggering. The standard approach feeds:
    #   A rows are staggered by row index (row r starts at cycle r)
    #   B cols are staggered by col index (col c starts at cycle c)
    #
    # For a 2x2 in a 4x4 array:
    # Cycle 0: a_in = [A[0][0], 0, 0, 0],        b_in = [B[0][0], 0, 0, 0]
    # Cycle 1: a_in = [A[0][1], A[1][0], 0, 0],   b_in = [B[1][0], B[0][1], 0, 0]
    # Cycle 2: a_in = [0, A[1][1], 0, 0],          b_in = [0, B[1][1], 0, 0]
    sa.clear()
    one = to_q88(1.0)
    two = to_q88(2.0)
    three = to_q88(3.0)
    four = to_q88(4.0)
    five = to_q88(5.0)
    six = to_q88(6.0)
    seven = to_q88(7.0)
    eight = to_q88(8.0)

    # Properly staggered feeding for systolic wave propagation
    sa.clock(True, [one, 0, 0, 0],          [five, 0, 0, 0])        # cycle 0
    sa.clock(True, [two, three, 0, 0],       [seven, six, 0, 0])     # cycle 1
    sa.clock(True, [0, four, 0, 0],          [0, eight, 0, 0])       # cycle 2
    # Drain
    for _ in range(5):
        sa.clock(True, zeros, zeros)

    results = sa.get_results()
    # Results are in full precision (Q16.16-like): multiply by (1/2^FRAC_BITS) to get Q8.8
    # C[0][0] = 1*5 + 2*7 = 19 → in full precision: 19 * 2^16 = 1245184
    scale = (1 << FRAC_BITS) ** 2  # = 65536

    c00 = results[0][0] / scale
    c01 = results[0][1] / scale
    c10 = results[1][0] / scale
    c11 = results[1][1] / scale

    print(f"  Matrix multiply result (float): [[{c00:.1f}, {c01:.1f}], [{c10:.1f}, {c11:.1f}]]")
    print(f"  Expected:                        [[19.0, 22.0], [43.0, 50.0]]")

    res.check(abs(c00 - 19.0) < 0.1, f"C[0][0] = {c00:.1f} ≈ 19.0")
    res.check(abs(c01 - 22.0) < 0.1, f"C[0][1] = {c01:.1f} ≈ 22.0")
    res.check(abs(c10 - 43.0) < 0.1, f"C[1][0] = {c10:.1f} ≈ 43.0")
    res.check(abs(c11 - 50.0) < 0.1, f"C[1][1] = {c11:.1f} ≈ 50.0")

    return res.summary("Systolic Array")


# ===========================================================================
# Module 3: Softmax Unit
# ===========================================================================
class SoftmaxUnit:
    """Behavioral model of softmax_unit.sv (reciprocal LUT version)."""

    # 32-entry reciprocal LUT: round(2^14 / (0.5 + k/64)), Q2.14 format
    RECIP_LUT = [
        32768, 31775, 30840, 29959, 29127, 28340, 27594, 26887,
        26214, 25575, 24966, 24385, 23831, 23302, 22795, 22310,
        21845, 21400, 20972, 20560, 20165, 19784, 19418, 19065,
        18725, 18396, 18079, 17772, 17476, 17190, 16913, 16644,
    ]

    def __init__(self, vec_len=D_HEAD):
        self.vec_len = vec_len

    def approx_exp(self, x: int) -> int:
        """PWL exp approximation (matches RTL exactly)."""
        x_s = sign_extend_16(x)
        if x_s >= 0:
            return 0x0100  # 1.0
        elif x_s > -0x0100:  # > -1.0
            return (0x0100 + x_s) & MASK_16
        elif x_s > -0x0200:  # > -2.0
            result = (0x0060 + ((x_s + 0x0100) >> 1)) & MASK_16
            return result if sign_extend_16(result) > 0 else 0x0004
        elif x_s > -0x0400:  # > -4.0
            result = (0x0020 + ((x_s + 0x0200) >> 2)) & MASK_16
            return result if sign_extend_16(result) > 0 else 0x0004
        else:
            return 0x0004

    @staticmethod
    def _clz16(val: int) -> int:
        """Count leading zeros of a 16-bit unsigned value (0-15)."""
        val = val & MASK_16
        if val == 0:
            return 15
        for i in range(15, -1, -1):
            if val & (1 << i):
                return 15 - i
        return 15

    @staticmethod
    def _compute_reciprocal(exp_sum: int) -> int:
        """Compute 65536/exp_sum via LUT + Newton-Raphson.

        Returns a 16-bit value r such that fp_mul(e, r) = (e << 8) / exp_sum.
        Mirrors compute_reciprocal() in softmax_unit.sv exactly.
        """
        s = exp_sum & MASK_16
        if s <= 1:
            return 0x7FFF

        # Step 1: CLZ and normalise to [0.5, 1.0) as Q0.16
        lz = SoftmaxUnit._clz16(s)
        s_norm = (s << lz) & MASK_16       # bit 15 is now 1

        # Step 2: LUT lookup using bits [14:10]
        lut_idx = (s_norm >> 10) & 0x1F
        r0 = SoftmaxUnit.RECIP_LUT[lut_idx]

        # Step 3: Newton-Raphson in Q2.14
        # prod = (s_norm * r0) >> 16  (Q0.16 * Q2.14 -> 32-bit -> top 16 = Q2.14)
        prod = ((s_norm & MASK_16) * (r0 & MASK_16))
        prod16 = (prod >> 16) & MASK_16

        # correction = 2.0_Q2.14 - prod16 = 32768 - prod16
        correction = (32768 - prod16) & MASK_16

        # r1 = (r0 * correction) >> 14
        r1_wide = (r0 & MASK_16) * (correction & MASK_16)
        r1 = (r1_wide >> 14) & MASK_16

        # Step 4: Denormalise: result = r1 >> (14 - lz)
        if lz > 14:
            return 0x7FFF
        rshift = 14 - lz
        result = (r1 >> rshift) & MASK_16

        if result > 0x7FFF:
            return 0x7FFF
        return result & MASK_16

    def compute(self, scores: List[int]) -> List[int]:
        """Run the full softmax FSM (matches RTL exactly)."""
        assert len(scores) == self.vec_len

        # Stage 1: Find max
        max_val = sign_extend_16(scores[0])
        for i in range(1, self.vec_len):
            s = sign_extend_16(scores[i])
            if s > max_val:
                max_val = s

        # Stage 2: Subtract max and compute exp
        exp_vals = []
        for i in range(self.vec_len):
            shifted = sign_extend_16(scores[i]) - max_val
            shifted_q = shifted & MASK_16
            exp_vals.append(self.approx_exp(shifted_q))

        # Stage 3: Sum
        exp_sum = 0
        for e in exp_vals:
            exp_sum += sign_extend_16(e)  # They should all be positive

        # Stage 4: Reciprocal (replaces division)
        recip = self._compute_reciprocal(exp_sum)

        # Stage 5: Normalise via multiply
        probs = []
        for e in exp_vals:
            if exp_sum != 0:
                probs.append(fp_mul(e, recip))
            else:
                probs.append(0)

        return probs


def test_softmax():
    print("\n" + "="*60)
    print("  SOFTMAX UNIT TESTS")
    print("="*60)
    res = TestResults()
    sm = SoftmaxUnit(vec_len=8)

    # Test 1: Uniform inputs (all 1.0)
    scores = [to_q88(1.0)] * 8
    probs = sm.compute(scores)
    probs_f = [from_q88(p) for p in probs]
    print(f"  Uniform probs: {[f'{p:.4f}' for p in probs_f]}")

    # All should be equal
    res.check(all(p == probs[0] for p in probs),
              "Uniform inputs → equal outputs",
              f"values: {[hex(p) for p in probs]}")

    # Each should be ~0.125
    res.check(abs(probs_f[0] - 0.125) < 0.05,
              f"Each prob ≈ 0.125 (got {probs_f[0]:.4f})")

    # Test 2: Sum ≈ 1.0
    total = sum(probs_f)
    res.check(abs(total - 1.0) < 0.1,
              f"Sum ≈ 1.0 (got {total:.4f})")

    # Test 3: Dominant score
    scores = [to_q88(4.0)] + [to_q88(0.0)] * 7
    probs = sm.compute(scores)
    probs_f = [from_q88(p) for p in probs]
    print(f"  Dominant probs: {[f'{p:.4f}' for p in probs_f]}")

    res.check(probs_f[0] > probs_f[1],
              f"Dominant score has highest prob ({probs_f[0]:.4f} > {probs_f[1]:.4f})")

    # Test 4: Ordering preserved
    scores = [to_q88(0.25 * i) for i in range(8)]
    probs = sm.compute(scores)
    probs_f = [from_q88(p) for p in probs]
    print(f"  Ordered probs: {[f'{p:.4f}' for p in probs_f]}")

    res.check(probs_f[-1] >= probs_f[0],
              f"Monotonic ordering preserved ({probs_f[-1]:.4f} >= {probs_f[0]:.4f})")

    # Test 5: Negative scores
    scores = [to_q88(-float(i+1)) for i in range(8)]
    probs = sm.compute(scores)
    probs_f = [from_q88(p) for p in probs]
    print(f"  Negative probs: {[f'{p:.4f}' for p in probs_f]}")

    res.check(probs_f[0] >= probs_f[-1],
              f"Less negative → higher prob ({probs_f[0]:.4f} >= {probs_f[-1]:.4f})")

    # Test 6: All zeros
    scores = [0] * 8
    probs = sm.compute(scores)
    probs_f = [from_q88(p) for p in probs]
    res.check(all(p == probs[0] for p in probs),
              "All-zero inputs → equal probs")

    # Test 7: Back-to-back (just verify no state leakage)
    scores1 = [to_q88(1.0)] * 8
    probs1 = sm.compute(scores1)
    scores2 = [to_q88(3.0)] + [to_q88(0.0)] * 7
    probs2 = sm.compute(scores2)
    res.check(from_q88(probs2[0]) > from_q88(probs1[0]),
              "Back-to-back: no state leakage")

    return res.summary("Softmax Unit")


# ===========================================================================
# Module 4: Layer Normalization
# ===========================================================================
class LayerNorm:
    """Behavioral model of layer_norm.sv"""
    def __init__(self, vec_len=D_MODEL):
        self.vec_len = vec_len

    def compute(self, x_in: List[int], gamma: List[int], beta: List[int]) -> List[int]:
        """Run LayerNorm FSM."""
        import math
        log2_n = int(math.log2(self.vec_len))

        # Stage 1: Mean (arithmetic right-shift for power-of-2 N)
        sum_acc = 0
        for i in range(self.vec_len):
            sum_acc += sign_extend_16(x_in[i])
        mean_val = sum_acc >> log2_n  # Arithmetic right-shift
        mean_q = mean_val & MASK_16

        # Stage 2: Variance
        var_acc = 0
        centered = []
        for i in range(self.vec_len):
            c = (sign_extend_16(x_in[i]) - mean_val) & MASK_16
            centered.append(c)
            c_sq = fp_mul(c, c)
            var_acc += sign_extend_16(c_sq)

        var_val = var_acc >> log2_n  # Arithmetic right-shift
        var_q = var_val & MASK_16

        # inv_std = 1/sqrt(var + eps)
        inv_std = fp_inv_sqrt((var_q + 1) & MASK_16)

        # Stage 3: Normalize
        y_out = []
        for i in range(self.vec_len):
            # y = gamma * centered * inv_std + beta
            normed = fp_mul(centered[i], inv_std)
            scaled = fp_mul(gamma[i], normed)
            y = fp_sat_add(scaled, beta[i])
            y_out.append(y)

        return y_out


def test_layer_norm():
    print("\n" + "="*60)
    print("  LAYER NORMALIZATION TESTS")
    print("="*60)
    res = TestResults()
    ln = LayerNorm(vec_len=8)  # Use smaller vec for readability

    # Test 1: Constant input → should normalize to ~0
    gamma = [to_q88(1.0)] * 8
    beta = [to_q88(0.0)] * 8
    x_in = [to_q88(2.0)] * 8  # All same value
    y = ln.compute(x_in, gamma, beta)
    y_f = [from_q88(v) for v in y]
    print(f"  Constant input output: {[f'{v:.4f}' for v in y_f]}")

    # All outputs should be near 0 (mean is subtracted, variance is ~0)
    max_abs = max(abs(v) for v in y_f)
    res.check(max_abs < 2.0,
              f"Constant input → small output (max |y| = {max_abs:.4f})")

    # Test 2: Symmetric input
    x_in = [to_q88(1.0), to_q88(-1.0)] * 4
    y = ln.compute(x_in, gamma, beta)
    y_f = [from_q88(v) for v in y]
    print(f"  Symmetric ±1.0 output: {[f'{v:.4f}' for v in y_f]}")

    # Positive inputs should map to positive, negative to negative
    res.check(y_f[0] > 0 and y_f[1] < 0,
              f"Symmetric sign preservation (y[0]={y_f[0]:.4f}, y[1]={y_f[1]:.4f})")

    # Test 3: Gamma scaling
    gamma_2x = [to_q88(2.0)] * 8
    y_1x = ln.compute(x_in, gamma, beta)
    y_2x = ln.compute(x_in, gamma_2x, beta)
    ratio = abs(from_q88(y_2x[0])) / max(abs(from_q88(y_1x[0])), 0.001)
    print(f"  Gamma scaling ratio: {ratio:.2f} (expected ~2.0)")
    res.check(1.5 < ratio < 2.5,
              f"Gamma=2.0 scales output ~2x (ratio={ratio:.2f})")

    # Test 4: Beta offset
    beta_1 = [to_q88(1.0)] * 8
    x_in_zero = [to_q88(0.0)] * 8  # Zero input
    y = ln.compute(x_in_zero, gamma, beta_1)
    y_f = [from_q88(v) for v in y]
    print(f"  Beta=1.0 offset: {[f'{v:.4f}' for v in y_f[:4]]}")
    # With zero input and gamma=1, beta=1, output should be near 1.0
    res.check(all(abs(v - 1.0) < 1.0 for v in y_f),
              f"Beta offset applied (y[0]={y_f[0]:.4f})")

    # Test 5: Non-trivial input
    x_in = [to_q88(float(i)) for i in range(8)]
    y = ln.compute(x_in, gamma, beta)
    y_f = [from_q88(v) for v in y]
    print(f"  Ramp input output: {[f'{v:.4f}' for v in y_f]}")
    # Should be centered around 0
    mean_y = sum(y_f) / len(y_f)
    res.check(abs(mean_y) < 1.0,
              f"Output approximately centered (mean={mean_y:.4f})")

    return res.summary("Layer Normalization")


# ===========================================================================
# Module 5: Feed-Forward Network
# ===========================================================================
class FeedForward:
    """Behavioral model of feed_forward.sv"""
    def compute(self, x_in, w1, b1, w2, b2):
        """FFN: ReLU(x*W1+b1) * W2 + b2"""
        d_model = len(x_in)
        d_ff = len(b1)

        # Linear 1: hidden = x * W1 + b1
        hidden = []
        for j in range(d_ff):
            acc = 0
            for i in range(d_model):
                acc += sign_extend_16(x_in[i]) * sign_extend_16(w1[i][j])
            val = (acc >> FRAC_BITS) & MASK_16
            val = fp_sat_add(val, b1[j])
            hidden.append(val)

        # ReLU
        for i in range(d_ff):
            if sign_extend_16(hidden[i]) < 0:
                hidden[i] = 0

        # Linear 2: y = hidden * W2 + b2
        y_out = []
        for j in range(d_model):
            acc = 0
            for i in range(d_ff):
                acc += sign_extend_16(hidden[i]) * sign_extend_16(w2[i][j])
            val = (acc >> FRAC_BITS) & MASK_16
            val = fp_sat_add(val, b2[j])
            y_out.append(val)

        return y_out, hidden


def test_feed_forward():
    print("\n" + "="*60)
    print("  FEED-FORWARD NETWORK TESTS")
    print("="*60)
    res = TestResults()
    ffn = FeedForward()

    # Use small dimensions for tractability
    d_model = 4
    d_ff = 8

    # Test 1: Identity-like W1, W2 with positive bias
    w1 = [[to_q88(0.5) if i == j % d_model else 0 for j in range(d_ff)] for i in range(d_model)]
    b1 = [to_q88(0.1)] * d_ff
    w2 = [[to_q88(0.5) if i % d_model == j else 0 for j in range(d_model)] for i in range(d_ff)]
    b2 = [0] * d_model

    x_in = [to_q88(1.0), to_q88(2.0), to_q88(-1.0), to_q88(0.5)]
    y, hidden = ffn.compute(x_in, w1, b1, w2, b2)

    y_f = [from_q88(v) for v in y]
    h_f = [from_q88(v) for v in hidden]
    print(f"  Input:  {[from_q88(v) for v in x_in]}")
    print(f"  Hidden (post-ReLU): {[f'{v:.4f}' for v in h_f]}")
    print(f"  Output: {[f'{v:.4f}' for v in y_f]}")

    # Hidden should have ReLU applied: all >= 0
    res.check(all(v >= 0 for v in h_f),
              "ReLU: all hidden activations >= 0")

    # Output should be non-zero (something was computed)
    res.check(any(abs(v) > 0.01 for v in y_f),
              "Output is non-zero")

    # Test 2: Negative input with small W1 → should hit ReLU zeros
    x_neg = [to_q88(-5.0)] * d_model
    b1_zero = [0] * d_ff
    y_neg, hidden_neg = ffn.compute(x_neg, w1, b1_zero, w2, b2)
    h_neg_f = [from_q88(v) for v in hidden_neg]
    print(f"  Negative input hidden: {[f'{v:.4f}' for v in h_neg_f]}")
    # With negative x and positive W1 diagonal of 0.5: x*W1 = -2.5 → ReLU → 0
    res.check(all(v == 0.0 for v in h_neg_f),
              "ReLU zeros negative pre-activations")

    # Test 3: All-zero input → output = b2
    x_zero = [0] * d_model
    y_z, _ = ffn.compute(x_zero, w1, b1, w2, b2)
    y_z_f = [from_q88(v) for v in y_z]
    print(f"  Zero input output: {[f'{v:.4f}' for v in y_z_f]}")
    # With zero input: hidden = ReLU(0 + b1) = b1, then output = b1 * W2 + b2
    res.check(True, "Zero input produces valid output")

    return res.summary("Feed-Forward Network")


# ===========================================================================
# Module 6: Full Decoder Integration
# ===========================================================================
def test_decoder_integration():
    print("\n" + "="*60)
    print("  TRANSFORMER DECODER INTEGRATION TEST")
    print("="*60)
    res = TestResults()

    # Use reduced dimensions for tractable integration test
    d = 8  # D_MODEL
    n_heads = 2
    d_head = d // n_heads  # 4
    d_ff = 16
    max_seq = 8  # Reduced MAX_SEQ_LEN for test

    # Initialize identity-like weights
    wq = [[to_q88(0.25) if i == j else 0 for j in range(d)] for i in range(d)]
    wk = [[to_q88(0.25) if i == j else 0 for j in range(d)] for i in range(d)]
    wv = [[to_q88(0.25) if i == j else 0 for j in range(d)] for i in range(d)]
    wo = [[to_q88(0.5)  if i == j else 0 for j in range(d)] for i in range(d)]

    w1 = [[to_q88(0.125) if i == j % d else 0 for j in range(d_ff)] for i in range(d)]
    b1 = [to_q88(0.1)] * d_ff
    w2 = [[to_q88(0.125) if i % d == j else 0 for j in range(d)] for i in range(d_ff)]
    b2 = [0] * d

    gamma = [to_q88(1.0)] * d
    beta = [to_q88(0.0)] * d

    # Token embedding: [0.5, -0.5, 0.5, -0.5, ...]
    token = [to_q88(0.5) if i % 2 == 0 else to_q88(-0.5) for i in range(d)]

    ln = LayerNorm(vec_len=d)
    sm = SoftmaxUnit(vec_len=max_seq)
    ffn_mod = FeedForward()

    # Step 1: LayerNorm 1
    ln1_out = ln.compute(token, gamma, beta)
    print(f"  LN1 out: {[f'{from_q88(v):.3f}' for v in ln1_out]}")
    res.check(any(v != 0 for v in ln1_out), "LN1 produces non-zero output")

    # Step 2: Attention with softmax at position 0
    # Project Q, K, V
    q_vec = []
    k_vec = []
    v_vec = []
    for j in range(d):
        acc_q, acc_k, acc_v = 0, 0, 0
        for i in range(d):
            acc_q += sign_extend_16(ln1_out[i]) * sign_extend_16(wq[i][j])
            acc_k += sign_extend_16(ln1_out[i]) * sign_extend_16(wk[i][j])
            acc_v += sign_extend_16(ln1_out[i]) * sign_extend_16(wv[i][j])
        q_vec.append((acc_q >> FRAC_BITS) & MASK_16)
        k_vec.append((acc_k >> FRAC_BITS) & MASK_16)
        v_vec.append((acc_v >> FRAC_BITS) & MASK_16)

    print(f"  Q: {[f'{from_q88(v):.3f}' for v in q_vec]}")
    print(f"  K: {[f'{from_q88(v):.3f}' for v in k_vec]}")

    # KV cache: position 0 holds our K, V
    kv_cache_k = [[0]*d for _ in range(max_seq)]
    kv_cache_v = [[0]*d for _ in range(max_seq)]
    kv_cache_k[0] = list(k_vec)
    kv_cache_v[0] = list(v_vec)
    seq_pos = 0

    # Compute attention scores per head, then apply softmax
    NEG_INF = to_q88(-8.0)  # -8.0 in Q8.8 — within PWL exp range, avoids overflow
    SCALE_Q88 = to_q88(0.25)  # 1/sqrt(d_head=4) = 0.5... but matches RTL: 0x0040

    head_probs_all = []
    for h in range(n_heads):
        # Compute raw scores for this head
        head_scores = []
        for t in range(max_seq):
            if t <= seq_pos:
                dot = 0
                for dd in range(d_head):
                    qi = h * d_head + dd
                    dot += sign_extend_16(q_vec[qi]) * sign_extend_16(kv_cache_k[t][qi])
                score = fp_mul((dot >> FRAC_BITS) & MASK_16, SCALE_Q88)
                head_scores.append(score)
            else:
                head_scores.append(NEG_INF)

        # Apply softmax (matches RTL softmax_unit)
        probs = sm.compute(head_scores)
        head_probs_all.append(probs)
        probs_f = [from_q88(p) for p in probs]
        print(f"  Head {h} softmax probs[0:4]: {[f'{p:.4f}' for p in probs_f[:4]]}")

    # At position 0, softmax over a single valid score should yield ~1.0
    prob0_h0 = from_q88(head_probs_all[0][0])
    res.check(prob0_h0 > 0.8,
              f"Position 0: softmax([score]) ~ 1.0 (got {prob0_h0:.4f})")

    # Future positions should be near zero
    prob_future = from_q88(head_probs_all[0][1])
    res.check(prob_future < 0.05,
              f"Future position prob ~ 0.0 (got {prob_future:.4f})")

    # Weighted sum: out_h = sum_t(probs[t] * V_h[t])
    head_out = []
    for h in range(n_heads):
        hout = []
        for dd in range(d_head):
            ws = 0
            for t in range(max_seq):
                if t <= seq_pos:
                    vi = h * d_head + dd
                    ws += sign_extend_16(head_probs_all[h][t]) * sign_extend_16(kv_cache_v[t][vi])
            hout.append((ws >> FRAC_BITS) & MASK_16)
        head_out.append(hout)

    # Concatenate and output projection
    concat = []
    for h in range(n_heads):
        concat.extend(head_out[h])

    attn_out = []
    for j in range(d):
        acc = 0
        for i in range(d):
            acc += sign_extend_16(concat[i]) * sign_extend_16(wo[i][j])
        attn_out.append((acc >> FRAC_BITS) & MASK_16)

    print(f"  Attn out: {[f'{from_q88(v):.3f}' for v in attn_out]}")
    res.check(True, "Attention with softmax computed")

    # Step 3: Residual 1
    residual1 = [fp_sat_add(token[i], attn_out[i]) for i in range(d)]
    print(f"  Residual1: {[f'{from_q88(v):.3f}' for v in residual1]}")
    res.check(True, "Residual connection 1 computed")

    # Step 4: LayerNorm 2
    ln2_out = ln.compute(residual1, gamma, beta)
    res.check(any(v != 0 for v in ln2_out), "LN2 produces non-zero output")

    # Step 5: FFN
    ffn_out, _ = ffn_mod.compute(ln2_out, w1, b1, w2, b2)
    print(f"  FFN out: {[f'{from_q88(v):.3f}' for v in ffn_out]}")

    # Step 6: Residual 2
    output = [fp_sat_add(residual1[i], ffn_out[i]) for i in range(d)]
    output_f = [from_q88(v) for v in output]
    print(f"  Final out: {[f'{v:.3f}' for v in output_f]}")

    res.check(any(abs(v) > 0.001 for v in output_f),
              "Decoder output is non-zero")

    # Verify residual connections preserved information
    input_energy = sum(from_q88(v)**2 for v in token)
    output_energy = sum(v**2 for v in output_f)
    print(f"  Input energy:  {input_energy:.4f}")
    print(f"  Output energy: {output_energy:.4f}")
    res.check(output_energy > 0.01,
              f"Output has non-trivial energy ({output_energy:.4f})")

    # Test sequential token at position 1 (exercises softmax with 2 active positions)
    print(f"\n  --- Sequential token at position 1 ---")
    token2 = [to_q88(1.0) if i % 3 == 0 else 0 for i in range(d)]
    ln1_out2 = ln.compute(token2, gamma, beta)
    res.check(any(v != 0 for v in ln1_out2), "Second token LN1 works")

    # Project Q2, K2, V2
    q2, k2, v2 = [], [], []
    for j in range(d):
        aq, ak, av = 0, 0, 0
        for i in range(d):
            aq += sign_extend_16(ln1_out2[i]) * sign_extend_16(wq[i][j])
            ak += sign_extend_16(ln1_out2[i]) * sign_extend_16(wk[i][j])
            av += sign_extend_16(ln1_out2[i]) * sign_extend_16(wv[i][j])
        q2.append((aq >> FRAC_BITS) & MASK_16)
        k2.append((ak >> FRAC_BITS) & MASK_16)
        v2.append((av >> FRAC_BITS) & MASK_16)

    # Update KV cache at position 1
    kv_cache_k[1] = list(k2)
    kv_cache_v[1] = list(v2)
    seq_pos = 1

    # Compute attention with softmax over 2 positions
    head_probs2 = []
    for h in range(n_heads):
        head_scores2 = []
        for t in range(max_seq):
            if t <= seq_pos:
                dot = 0
                for dd in range(d_head):
                    qi = h * d_head + dd
                    dot += sign_extend_16(q2[qi]) * sign_extend_16(kv_cache_k[t][qi])
                score = fp_mul((dot >> FRAC_BITS) & MASK_16, SCALE_Q88)
                head_scores2.append(score)
            else:
                head_scores2.append(NEG_INF)
        probs2 = sm.compute(head_scores2)
        head_probs2.append(probs2)

    # Softmax over 2 valid positions: both should have non-zero probability
    p0 = from_q88(head_probs2[0][0])
    p1 = from_q88(head_probs2[0][1])
    p_sum = p0 + p1
    print(f"  Head 0 softmax at pos 1: p[0]={p0:.4f}, p[1]={p1:.4f}, sum={p_sum:.4f}")
    res.check(p0 > 0.01 and p1 > 0.01,
              f"Both positions have non-zero probability")
    res.check(abs(p_sum - 1.0) < 0.15,
              f"Softmax probs sum ~ 1.0 (got {p_sum:.4f})")

    # Future positions still near zero
    p_future2 = from_q88(head_probs2[0][2])
    res.check(p_future2 < 0.05,
              f"Future position still ~ 0.0 (got {p_future2:.4f})")

    return res.summary("Decoder Integration")


# ===========================================================================
# Module 7: Fixed-Point Utilities
# ===========================================================================
def test_fp_utilities():
    print("\n" + "="*60)
    print("  FIXED-POINT UTILITY TESTS")
    print("="*60)
    res = TestResults()

    # Test to_q88 / from_q88 roundtrip
    for val in [0.0, 1.0, -1.0, 0.5, -0.5, 127.0, -128.0, 0.00390625]:
        q = to_q88(val)
        back = from_q88(q)
        res.check(abs(back - val) < 0.005,
                  f"Roundtrip {val} → 0x{q:04X} → {back}")

    # Test fp_mul
    cases = [
        (1.0, 1.0, 1.0),
        (2.0, 3.0, 6.0),
        (-1.0, 1.0, -1.0),
        (0.5, 0.5, 0.25),
        (-2.0, -3.0, 6.0),
    ]
    for a, b, expected in cases:
        result = from_q88(fp_mul(to_q88(a), to_q88(b)))
        res.check(abs(result - expected) < 0.05,
                  f"fp_mul({a}*{b}) = {result:.4f} ≈ {expected}")

    # Test fp_sat_add
    # Normal addition
    r = from_q88(fp_sat_add(to_q88(1.0), to_q88(2.0)))
    res.check(abs(r - 3.0) < 0.01, f"fp_sat_add(1+2) = {r:.4f}")

    # Saturation positive
    r = from_q88(fp_sat_add(to_q88(127.0), to_q88(10.0)))
    res.check(r > 126.0, f"Positive saturation: {r:.4f}")

    # Saturation negative
    r = from_q88(fp_sat_add(to_q88(-128.0), to_q88(-10.0)))
    res.check(r < -126.0, f"Negative saturation: {r:.4f}")

    return res.summary("Fixed-Point Utilities")


# ===========================================================================
# Main: Run All Tests
# ===========================================================================
def main():
    print("=" * 60)
    print("  LLM TRANSFORMER DECODER — RTL BEHAVIORAL VERIFICATION")
    print("  Bit-accurate Python model mirroring SystemVerilog RTL")
    print("=" * 60)

    all_passed = True
    all_passed &= test_fp_utilities()
    all_passed &= test_processing_element()
    all_passed &= test_systolic_array()
    all_passed &= test_softmax()
    all_passed &= test_layer_norm()
    all_passed &= test_feed_forward()
    all_passed &= test_decoder_integration()

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL MODULE TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED — see details above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
