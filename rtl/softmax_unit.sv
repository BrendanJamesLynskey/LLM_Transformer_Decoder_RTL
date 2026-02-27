// =============================================================================
// softmax_unit.sv - Piecewise-Linear Softmax Approximation
// =============================================================================
// Computes an approximate softmax over a vector of fixed-point scores.
// Uses the max-subtract-exp-normalize approach:
//   1. Find max(x_i) across the input vector
//   2. Compute shifted = x_i - max for numerical stability
//   3. Approximate exp(shifted) using piecewise linear segments
//   4. Sum all exp values
//   5. Compute reciprocal of sum via 32-entry LUT + Newton-Raphson
//   6. Normalize: out_i = exp(shifted_i) * recip(sum)
//
// Division Replacement:
//   The normalisation step probs[i] = exp[i] / exp_sum is replaced by:
//     recip = 65536 / exp_sum   (computed via LUT + Newton-Raphson)
//     probs[i] = fp_mul(exp[i], recip)
//   where fp_mul(a, b) = (a * b) >>> FRAC_BITS.
//
//   This works because fp_mul interprets both operands as Q8.8:
//     (exp[i] * recip) >> 8 = (exp[i] << 8) / exp_sum
//   which is the original fixed-point division formula.
//
//   The reciprocal is computed in a single cycle using:
//     - CLZ (count leading zeros) to normalise exp_sum to [0.5, 1.0)
//     - 32-entry LUT indexed by top 5 mantissa bits (Q2.14 format)
//     - One Newton-Raphson iteration: r' = r * (2 - x_norm * r)
//     - Denormalising shift: recip = r' >> (14 - lz)
//
//   Accuracy: ~12 bits after Newton-Raphson, sufficient for Q8.8 output.
//   Area: 32x16-bit LUT + one 16x16 multiplier + one 32x16 multiplier.
//   Latency: 1 cycle (fully combinational function).
// =============================================================================

module softmax_unit
  import transformer_pkg::*;
#(
  parameter int VEC_LEN = D_HEAD  // Length of input vector
)(
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  start,
  input  data_t                 scores [VEC_LEN],

  output data_t                 probs  [VEC_LEN],
  output logic                  valid
);

  // =========================================================================
  // State Machine
  // =========================================================================
  typedef enum logic [2:0] {
    S_IDLE,
    S_FIND_MAX,
    S_SUBTRACT_EXP,
    S_SUM,
    S_RECIP,        // Compute reciprocal of exp_sum (1 cycle)
    S_NORMALIZE,    // Multiply each exp by reciprocal
    S_DONE
  } state_t;

  state_t state, state_next;

  // Internal registers
  data_t max_val;
  data_t shifted  [VEC_LEN];
  data_t exp_vals [VEC_LEN];
  acc_t  exp_sum;
  logic [$clog2(VEC_LEN):0] idx;

  // Reciprocal of exp_sum (computed in S_RECIP, used in S_NORMALIZE)
  data_t recip_val;

  // =========================================================================
  // Piecewise-Linear Exponential Approximation (for negative inputs)
  // =========================================================================
  // exp(x) for x in [-8, 0] mapped to Q8.8
  // Segments: [-8,-4]: ~0, [-4,-2]: steep ramp, [-2,-1]: moderate, [-1,0]: ~1
  function automatic data_t approx_exp(input data_t x);
    if (x >= 0)
      return 16'sh0100;           // exp(0) = 1.0
    else if (x > -16'sh0100)      // x in (-1, 0)
      return 16'sh0100 + x;       // Linear: 1 + x ~ exp(x) near 0
    else if (x > -16'sh0200)      // x in (-2, -1)
      return 16'sh0060 + (x + 16'sh0100) >>> 1; // ~0.375 + 0.5*(x+1)
    else if (x > -16'sh0400)      // x in (-4, -2)
      return 16'sh0020 + (x + 16'sh0200) >>> 2; // ~0.125 + 0.25*(x+2)
    else
      return 16'sh0004;           // ~0.015 for very negative
  endfunction

  // =========================================================================
  // 32-Entry Reciprocal LUT (Q2.14 format)
  // =========================================================================
  // Entry k stores round(2^14 / (0.5 + k/64)).
  // Input x_norm is in [0.5, 1.0); index = top 5 bits of mantissa.
  // Output is 1/x_norm in Q2.14 (range [1.0, 2.0]).
  function automatic logic [15:0] recip_lut(input logic [4:0] index);
    case (index)
      5'd0:  return 16'd32768; // 1/0.5000 = 2.0000
      5'd1:  return 16'd31775; // 1/0.5156 = 1.9394
      5'd2:  return 16'd30840; // 1/0.5312 = 1.8824
      5'd3:  return 16'd29959; // 1/0.5469 = 1.8286
      5'd4:  return 16'd29127; // 1/0.5625 = 1.7778
      5'd5:  return 16'd28340; // 1/0.5781 = 1.7297
      5'd6:  return 16'd27594; // 1/0.5938 = 1.6842
      5'd7:  return 16'd26887; // 1/0.6094 = 1.6410
      5'd8:  return 16'd26214; // 1/0.6250 = 1.6000
      5'd9:  return 16'd25575; // 1/0.6406 = 1.5610
      5'd10: return 16'd24966; // 1/0.6562 = 1.5238
      5'd11: return 16'd24385; // 1/0.6719 = 1.4884
      5'd12: return 16'd23831; // 1/0.6875 = 1.4545
      5'd13: return 16'd23302; // 1/0.7031 = 1.4222
      5'd14: return 16'd22795; // 1/0.7188 = 1.3913
      5'd15: return 16'd22310; // 1/0.7344 = 1.3617
      5'd16: return 16'd21845; // 1/0.7500 = 1.3333
      5'd17: return 16'd21400; // 1/0.7656 = 1.3061
      5'd18: return 16'd20972; // 1/0.7812 = 1.2800
      5'd19: return 16'd20560; // 1/0.7969 = 1.2549
      5'd20: return 16'd20165; // 1/0.8125 = 1.2308
      5'd21: return 16'd19784; // 1/0.8281 = 1.2076
      5'd22: return 16'd19418; // 1/0.8438 = 1.1852
      5'd23: return 16'd19065; // 1/0.8594 = 1.1636
      5'd24: return 16'd18725; // 1/0.8750 = 1.1429
      5'd25: return 16'd18396; // 1/0.8906 = 1.1228
      5'd26: return 16'd18079; // 1/0.9062 = 1.1035
      5'd27: return 16'd17772; // 1/0.9219 = 1.0847
      5'd28: return 16'd17476; // 1/0.9375 = 1.0667
      5'd29: return 16'd17190; // 1/0.9531 = 1.0492
      5'd30: return 16'd16913; // 1/0.9688 = 1.0323
      5'd31: return 16'd16644; // 1/0.9844 = 1.0159
    endcase
  endfunction

  // =========================================================================
  // Count Leading Zeros (16-bit)
  // =========================================================================
  function automatic logic [3:0] clz16(input logic [15:0] val);
    for (int i = 15; i >= 0; i--) begin
      if (val[i]) return 4'(15 - i);
    end
    return 4'd15;
  endfunction

  // =========================================================================
  // Reciprocal Computation: 65536 / exp_sum
  // =========================================================================
  // Returns a 16-bit value r such that fp_mul(exp_val, r) = (exp_val<<8)/exp_sum.
  //
  // Algorithm:
  //   1. Normalise: s_norm = exp_sum << lz, so MSB is bit 15
  //   2. LUT: r0 = 1/s_norm in Q2.14, indexed by bits [14:10]
  //   3. Newton-Raphson: r1 = r0 * (2 - s_norm * r0), all in Q2.14
  //   4. Denormalise: result = r1 >> (14 - lz)
  //
  // The result represents 65536/exp_sum as a raw 16-bit integer. When used
  // as the second operand of fp_mul (which interprets it as Q8.8):
  //   fp_mul(e, r) = (e * r) >> 8 = e * (65536/S) >> 8 = (e << 8) / S
  // which is exactly the original division formula.
  function automatic data_t compute_reciprocal(input logic [15:0] s);
    logic [3:0]  lz;
    logic [15:0] s_norm;
    logic [4:0]  lut_idx;
    logic [15:0] r0;
    logic [31:0] prod;
    logic [15:0] prod16;
    logic [15:0] correction;
    logic [31:0] r1_wide;
    logic [15:0] r1;
    logic [3:0]  rshift;
    logic [15:0] result;

    if (s <= 16'd1) return 16'sh7FFF;  // Clamp: 1/0 or 1/1 -> max positive

    // Step 1: Count leading zeros, normalise
    lz = clz16(s);
    s_norm = s << lz;              // Bit 15 is now 1; [0.5, 1.0) in Q0.16

    // Step 2: LUT lookup using bits [14:10] (5 bits after leading 1)
    lut_idx = s_norm[14:10];
    r0 = recip_lut(lut_idx);       // ~1/s_norm in Q2.14

    // Step 3: Newton-Raphson iteration in Q2.14
    //   prod = s_norm * r0: Q0.16 * Q2.14 = 32-bit; >> 16 gives Q2.14
    //   correction = 2.0_Q2.14 - prod = 32768 - prod
    //   r1 = (r0 * correction) >> 14
    prod = {16'b0, s_norm} * {16'b0, r0};
    prod16 = prod[31:16];           // Q2.14 truncation
    correction = 16'd32768 - prod16; // 2.0 in Q2.14 = 32768
    r1_wide = {16'b0, r0} * {16'b0, correction};
    r1 = r1_wide[29:14];           // >> 14, keep 16 bits

    // Step 4: Denormalise
    //   r1 ≈ 2^30 / s_norm (as integer). We want 65536/s = 2^16/s.
    //   Since s = s_norm >> lz: 2^16/s = 2^(16+lz)/s_norm = r1 * 2^(lz-14)
    //   So: result = r1 >> (14 - lz)
    rshift = 4'd14 - lz;
    // lz ranges 0..14 for valid 16-bit inputs, so rshift is 0..14
    // For very small sums (large lz), rshift could be negative (left shift),
    // but those cases will saturate anyway.
    if (lz > 4'd14) begin
      // exp_sum < 2: result would overflow Q8.8, clamp
      result = 16'sh7FFF;
    end else begin
      result = r1 >> rshift;
    end

    // Clamp to max positive Q8.8
    if (result > 16'h7FFF)
      return 16'sh7FFF;
    else
      return data_t'(result);
  endfunction

  // =========================================================================
  // FSM Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      max_val   <= '0;
      exp_sum   <= '0;
      idx       <= '0;
      valid     <= 1'b0;
      recip_val <= '0;
      for (int i = 0; i < VEC_LEN; i++) begin
        shifted[i]  <= '0;
        exp_vals[i] <= '0;
        probs[i]    <= '0;
      end
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            max_val <= scores[0];
            idx     <= 1;
          end
        end

        S_FIND_MAX: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            if (scores[idx] > max_val)
              max_val <= scores[idx];
            idx <= idx + 1;
          end else begin
            idx <= '0;
          end
        end

        S_SUBTRACT_EXP: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            shifted[idx]  <= scores[idx] - max_val;
            exp_vals[idx] <= approx_exp(scores[idx] - max_val);
            idx <= idx + 1;
          end else begin
            idx     <= '0;
            exp_sum <= '0;
          end
        end

        S_SUM: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            exp_sum <= exp_sum + acc_t'(exp_vals[idx]);
            idx <= idx + 1;
          end else begin
            idx <= '0;
          end
        end

        // Compute reciprocal once (replaces per-element division)
        S_RECIP: begin
          recip_val <= compute_reciprocal(exp_sum[15:0]);
          idx       <= '0;
        end

        // Normalise: probs[i] = fp_mul(exp_vals[i], recip_val)
        S_NORMALIZE: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            if (exp_sum != 0)
              probs[idx] <= fp_mul(exp_vals[idx], recip_val);
            else
              probs[idx] <= '0;
            idx <= idx + 1;
          end else begin
            valid <= 1'b1;
          end
        end

        S_DONE: begin
          // Hold valid until next start
        end

        default: ;
      endcase
    end
  end

  // =========================================================================
  // FSM Next-State Logic
  // =========================================================================
  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:         if (start) state_next = S_FIND_MAX;
      S_FIND_MAX:     if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_SUBTRACT_EXP;
      S_SUBTRACT_EXP: if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_SUM;
      S_SUM:          if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_RECIP;
      S_RECIP:        state_next = S_NORMALIZE;
      S_NORMALIZE:    if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_DONE;
      S_DONE:         state_next = S_IDLE;
      default:        state_next = S_IDLE;
    endcase
  end

endmodule
