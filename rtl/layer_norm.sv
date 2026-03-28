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

  output reg signed [VEC_LEN-1:0][DATA_WIDTH-1:0] y_out,  // Normalized output
  output reg   valid
);

  // Compile-time check: mean/variance computation uses arithmetic right-shift
  // (>>> $clog2(VEC_LEN)) which is only correct for power-of-2 lengths.
  initial begin
    if (VEC_LEN & (VEC_LEN - 1))
      $fatal(1, "layer_norm: VEC_LEN=%0d is not a power of 2; right-shift division is incorrect for non-power-of-2 lengths", VEC_LEN);
  end

  integer i;
  localparam [2:0] S_IDLE = 0;
  localparam [2:0] S_MEAN = 1;
  localparam [2:0] S_VARIANCE = 2;
  localparam [2:0] S_NORMALIZE = 3;
  localparam [2:0] S_DONE = 4;

  reg [2:0] state, state_next;

  reg signed [31:0]   sum_acc;
  reg signed [31:0]   var_acc;
  reg signed [15:0]  mean_val;
  reg signed [15:0]  var_val;
  reg signed [15:0]  inv_std;
  logic [$clog2(VEC_LEN):0] idx;

  // Intermediate: x - mean
  reg signed [15:0] centered [VEC_LEN];

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      sum_acc  <= 0;
      var_acc  <= 0;
      mean_val <= 0;
      var_val  <= 0;
      inv_std  <= 0;
      idx      <= 0;
      valid    <= 1'b0;
      for (i = 0; i < VEC_LEN; i = i + 1) begin
        y_out[i]    <= 0;
        centered[i] <= 0;
      end
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            sum_acc <= 0;
            var_acc <= 0;
            idx     <= 0;
          end
        end

        S_MEAN: begin
          if (idx < VEC_LEN) begin
            sum_acc <= sum_acc + to_acc(x_in[idx]);
            idx <= idx + 1;
          end else begin
            // mean = sum / N via arithmetic right-shift (VEC_LEN is power of 2)
            mean_val <= to_data(sum_acc >>> $clog2(VEC_LEN));
            idx      <= 0;
          end
        end

        S_VARIANCE: begin
          if (idx < VEC_LEN) begin
            centered[idx] <= x_in[idx] - mean_val;
            var_acc <= var_acc + to_acc(fp_mul(x_in[idx] - mean_val, x_in[idx] - mean_val));
            idx <= idx + 1;
          end else begin
            var_val <= to_data(var_acc >>> $clog2(VEC_LEN));
            // Compute 1/sqrt(variance + epsilon) via LUT + Newton-Raphson
            inv_std <= fp_inv_sqrt(to_data(var_acc >>> $clog2(VEC_LEN)) + 16'sh0001);
            idx     <= 0;
          end
        end

        S_NORMALIZE: begin
          if (idx < VEC_LEN) begin
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
  always @(*) begin
    state_next = state;
    case (state)
      S_IDLE:      if (start) state_next = S_MEAN;
      S_MEAN:      if (idx >= VEC_LEN) state_next = S_VARIANCE;
      S_VARIANCE:  if (idx >= VEC_LEN) state_next = S_NORMALIZE;
      S_NORMALIZE: if (idx >= VEC_LEN) state_next = S_DONE;
      S_DONE:      state_next = S_IDLE;
      default:     state_next = S_IDLE;
    endcase
  end

endmodule
