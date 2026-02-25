// =============================================================================
// layer_norm.sv - Layer Normalization Unit
// =============================================================================
// Implements LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
//
// Pipeline stages:
//   1. Compute mean = sum(x_i) / N
//   2. Compute variance = sum((x_i - mean)^2) / N
//   3. Normalize: y_i = gamma * (x_i - mean) * rsqrt(var + eps) + beta
//
// Uses fixed-point arithmetic throughout. The reciprocal sqrt is approximated
// via a small LUT defined in transformer_pkg.
// =============================================================================

module layer_norm
  import transformer_pkg::*;
#(
  parameter int VEC_LEN = D_MODEL
)(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  input  data_t  x_in    [VEC_LEN],  // Input vector
  input  data_t  gamma   [VEC_LEN],  // Scale parameters
  input  data_t  beta    [VEC_LEN],  // Bias parameters

  output data_t  y_out   [VEC_LEN],  // Normalized output
  output logic   valid
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_MEAN,
    S_VARIANCE,
    S_NORMALIZE,
    S_DONE
  } state_t;

  state_t state, state_next;

  acc_t   sum_acc;
  acc_t   var_acc;
  data_t  mean_val;
  data_t  var_val;
  data_t  inv_std;
  logic [$clog2(VEC_LEN):0] idx;

  // Intermediate: x - mean
  data_t centered [VEC_LEN];

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      sum_acc  <= '0;
      var_acc  <= '0;
      mean_val <= '0;
      var_val  <= '0;
      inv_std  <= '0;
      idx      <= '0;
      valid    <= 1'b0;
      for (int i = 0; i < VEC_LEN; i++) begin
        y_out[i]    <= '0;
        centered[i] <= '0;
      end
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            sum_acc <= '0;
            var_acc <= '0;
            idx     <= '0;
          end
        end

        S_MEAN: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            sum_acc <= sum_acc + acc_t'(x_in[idx]);
            idx <= idx + 1;
          end else begin
            // mean = sum / N (shift for power-of-2 N, else divide)
            mean_val <= data_t'(sum_acc / VEC_LEN);
            idx      <= '0;
          end
        end

        S_VARIANCE: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            centered[idx] <= x_in[idx] - mean_val;
            var_acc <= var_acc + acc_t'(fp_mul(x_in[idx] - mean_val, x_in[idx] - mean_val));
            idx <= idx + 1;
          end else begin
            var_val <= data_t'(var_acc / VEC_LEN);
            // Compute 1/sqrt(variance + epsilon)
            inv_std <= fp_inv_sqrt(data_t'(var_acc / VEC_LEN) + 16'sh0001);
            idx     <= '0;
          end
        end

        S_NORMALIZE: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            // y = gamma * (x - mean) * inv_std + beta
            y_out[idx] <= fp_sat_add(
              fp_mul(gamma[idx], fp_mul(centered[idx], inv_std)),
              beta[idx]
            );
            idx <= idx + 1;
          end else begin
            valid <= 1'b1;
          end
        end

        S_DONE: ;

        default: ;
      endcase
    end
  end

  // =========================================================================
  // Next-State Logic
  // =========================================================================
  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:      if (start) state_next = S_MEAN;
      S_MEAN:      if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_VARIANCE;
      S_VARIANCE:  if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_NORMALIZE;
      S_NORMALIZE: if (idx >= VEC_LEN[$clog2(VEC_LEN):0]) state_next = S_DONE;
      S_DONE:      state_next = S_IDLE;
      default:     state_next = S_IDLE;
    endcase
  end

endmodule
