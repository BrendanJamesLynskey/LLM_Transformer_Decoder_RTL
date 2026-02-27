// =============================================================================
// feed_forward.sv - Position-wise Feed-Forward Network
// =============================================================================
// Implements FFN(x) = ReLU(x * W1 + b1) * W2 + b2
//
// Two linear layers with ReLU activation between them.
// Inner dimension D_FF is typically 4x the model dimension.
// Processes one token at a time during autoregressive inference.
// =============================================================================

module feed_forward
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] x_in,

  // Weights and biases (unpacked 2D — set before sim)
  input  data_t  w1 [D_MODEL][D_FF],
  input  data_t  b1 [D_FF],
  input  data_t  w2 [D_FF][D_MODEL],
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2,

  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] y_out,
  output logic   valid
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_LINEAR1,
    S_RELU,
    S_LINEAR2,
    S_DONE
  } state_t;

  state_t state, state_next;

  data_t hidden [D_FF];
  logic [$clog2(D_FF):0]    ff_idx;
  logic [$clog2(D_MODEL):0] out_idx;

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      valid   <= 1'b0;
      ff_idx  <= '0;
      out_idx <= '0;
      for (int i = 0; i < D_FF; i++)
        hidden[i] <= '0;
      for (int i = 0; i < D_MODEL; i++)
        y_out[i] <= '0;
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            ff_idx  <= '0;
            out_idx <= '0;
          end
        end

        // First linear layer: hidden = x * W1 + b1
        S_LINEAR1: begin
          if (ff_idx < D_FF[$clog2(D_FF):0]) begin
            acc_t acc = '0;
            for (int j = 0; j < D_MODEL; j++)
              acc = acc + acc_t'(x_in[j]) * acc_t'(w1[j][ff_idx]);
            hidden[ff_idx] <= data_t'(acc >>> FRAC_BITS) + b1[ff_idx];
            ff_idx <= ff_idx + 1;
          end
        end

        // ReLU activation
        S_RELU: begin
          for (int i = 0; i < D_FF; i++) begin
            if (hidden[i][DATA_WIDTH-1]) // Negative (sign bit set)
              hidden[i] <= '0;
          end
          out_idx <= '0;
        end

        // Second linear layer: y = hidden * W2 + b2
        S_LINEAR2: begin
          if (out_idx < D_MODEL[$clog2(D_MODEL):0]) begin
            acc_t acc = '0;
            for (int j = 0; j < D_FF; j++)
              acc = acc + acc_t'(hidden[j]) * acc_t'(w2[j][out_idx]);
            y_out[out_idx] <= data_t'(acc >>> FRAC_BITS) + b2[out_idx];
            out_idx <= out_idx + 1;
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
      S_IDLE:    if (start) state_next = S_LINEAR1;
      S_LINEAR1: if (ff_idx >= D_FF[$clog2(D_FF):0]) state_next = S_RELU;
      S_RELU:    state_next = S_LINEAR2;
      S_LINEAR2: if (out_idx >= D_MODEL[$clog2(D_MODEL):0]) state_next = S_DONE;
      S_DONE:    state_next = S_IDLE;
      default:   state_next = S_IDLE;
    endcase
  end

endmodule
