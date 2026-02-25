// =============================================================================
// softmax_unit.sv - Piecewise-Linear Softmax Approximation
// =============================================================================
// Computes an approximate softmax over a vector of fixed-point scores.
// Uses the max-subtract-exp-normalize approach:
//   1. Find max(x_i) across the input vector
//   2. Compute shifted = x_i - max for numerical stability
//   3. Approximate exp(shifted) using piecewise linear segments
//   4. Normalize: out_i = exp(shifted_i) / sum(exp(shifted_j))
//
// The exponential is approximated by a 4-segment PWL function operating
// on the Q8.8 fixed-point format.
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
    S_NORMALIZE,
    S_DONE
  } state_t;

  state_t state, state_next;

  // Internal registers
  data_t max_val;
  data_t shifted  [VEC_LEN];
  data_t exp_vals [VEC_LEN];
  acc_t  exp_sum;
  logic [$clog2(VEC_LEN):0] idx;

  // =========================================================================
  // Piecewise-Linear Exponential Approximation (for negative inputs)
  // =========================================================================
  // exp(x) for x in [-8, 0] mapped to Q8.8
  // Segments: [-8,-4]: ~0, [-4,-2]: steep ramp, [-2,-1]: moderate, [-1,0]: ~1
  function automatic data_t approx_exp(input data_t x);
    if (x >= 0)
      return 16'sh0100;           // exp(0) = 1.0
    else if (x > -16'sh0100)      // x in (-1, 0)
      return 16'sh0100 + x;       // Linear: 1 + x ≈ exp(x) near 0
    else if (x > -16'sh0200)      // x in (-2, -1)
      return 16'sh0060 + (x + 16'sh0100) >>> 1; // ~0.375 + 0.5*(x+1)
    else if (x > -16'sh0400)      // x in (-4, -2)
      return 16'sh0020 + (x + 16'sh0200) >>> 2; // ~0.125 + 0.25*(x+2)
    else
      return 16'sh0004;           // ~0.015 for very negative
  endfunction

  // =========================================================================
  // FSM Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      max_val <= '0;
      exp_sum <= '0;
      idx     <= '0;
      valid   <= 1'b0;
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

        S_NORMALIZE: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            // probs[i] = exp_vals[i] / exp_sum
            // Fixed-point division: (exp * 2^FRAC) / sum
            if (exp_sum != 0)
              probs[idx] <= data_t'((acc_t'(exp_vals[idx]) <<< FRAC_BITS) / exp_sum);
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
      S_SUM:          if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_NORMALIZE;
      S_NORMALIZE:    if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_DONE;
      S_DONE:         state_next = S_IDLE;
      default:        state_next = S_IDLE;
    endcase
  end

endmodule
