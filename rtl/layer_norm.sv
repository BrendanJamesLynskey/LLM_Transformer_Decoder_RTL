// =============================================================================
// layer_norm.sv - Layer Normalization Unit
// =============================================================================
// Implements LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
//
// Pipeline stages:
//   1. Compute mean = sum(x_i) >>> log2(N)  (arithmetic right-shift)
//   2. Compute variance = sum((x_i - mean)^2) >>> log2(N)
//   3. Compute inv_std = 1/sqrt(var + eps) via 32-entry rsqrt LUT + Newton-Raphson
//   4. Normalize: y_i = gamma * (x_i - mean) * inv_std + beta
//
// Division-free: all divisions by VEC_LEN use arithmetic right-shift (VEC_LEN
// must be a power of 2). The reciprocal square root uses a 32-entry LUT with
// one Newton-Raphson iteration (~12 bits accuracy), matching the approach
// proven in the softmax normalisation path.
// =============================================================================

module layer_norm
  import transformer_pkg::*;
#(
  parameter int VEC_LEN = D_MODEL
)(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  input  logic signed [VEC_LEN-1:0][DATA_WIDTH-1:0] x_in,   // Input vector
  input  logic signed [VEC_LEN-1:0][DATA_WIDTH-1:0] gamma,  // Scale parameters
  input  logic signed [VEC_LEN-1:0][DATA_WIDTH-1:0] beta,   // Bias parameters

  output logic signed [VEC_LEN-1:0][DATA_WIDTH-1:0] y_out,  // Normalized output
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
            // mean = sum / N via arithmetic right-shift (VEC_LEN is power of 2)
            mean_val <= data_t'(sum_acc >>> $clog2(VEC_LEN));
            idx      <= '0;
          end
        end

        S_VARIANCE: begin
          if (idx < VEC_LEN[$clog2(VEC_LEN):0]) begin
            centered[idx] <= x_in[idx] - mean_val;
            var_acc <= var_acc + acc_t'(fp_mul(x_in[idx] - mean_val, x_in[idx] - mean_val));
            idx <= idx + 1;
          end else begin
            var_val <= data_t'(var_acc >>> $clog2(VEC_LEN));
            // Compute 1/sqrt(variance + epsilon) via LUT + Newton-Raphson
            inv_std <= fp_inv_sqrt(data_t'(var_acc >>> $clog2(VEC_LEN)) + 16'sh0001);
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
