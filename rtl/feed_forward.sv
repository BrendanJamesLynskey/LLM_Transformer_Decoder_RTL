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
  input  signed [15:0]  w1 [D_MODEL][D_FF],
  input  signed [15:0]  b1 [D_FF],
  input  signed [15:0]  w2 [D_FF][D_MODEL],
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2,

  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] y_out,
  output reg   valid
);

  integer i, j;
  reg signed [31:0] acc;
  localparam [2:0] S_IDLE = 0;
  localparam [2:0] S_LINEAR1 = 1;
  localparam [2:0] S_RELU = 2;
  localparam [2:0] S_LINEAR2 = 3;
  localparam [2:0] S_DONE = 4;

  reg [2:0] state, state_next;

  reg signed [15:0] hidden [D_FF];
  logic [$clog2(D_FF):0]    ff_idx;
  logic [$clog2(D_MODEL):0] out_idx;

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      valid   <= 1'b0;
      ff_idx  <= 0;
      out_idx <= 0;
      for (i = 0; i < D_FF; i = i + 1)
        hidden[i] <= 0;
      for (i = 0; i < D_MODEL; i = i + 1)
        y_out[i] <= 0;
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            ff_idx  <= 0;
            out_idx <= 0;
          end
        end

        // First linear layer: hidden = x * W1 + b1
        S_LINEAR1: begin
          if (ff_idx < D_FF) begin
            acc = 0;
            for (j = 0; j < D_MODEL; j = j + 1)
              acc = acc + to_acc(x_in[j]) * to_acc(w1[j][ff_idx]);
            hidden[ff_idx] <= to_data(acc >>> FRAC_BITS) + b1[ff_idx];
            ff_idx <= ff_idx + 1;
          end
        end

        // ReLU activation
        S_RELU: begin
          for (i = 0; i < D_FF; i = i + 1) begin
            if (hidden[i][DATA_WIDTH-1]) // Negative (sign bit set)
              hidden[i] <= 0;
          end
          out_idx <= 0;
        end

        // Second linear layer: y = hidden * W2 + b2
        S_LINEAR2: begin
          if (out_idx < D_MODEL) begin
            acc = 0;
            for (j = 0; j < D_FF; j = j + 1)
              acc = acc + to_acc(hidden[j]) * to_acc(w2[j][out_idx]);
            y_out[out_idx] <= to_data(acc >>> FRAC_BITS) + b2[out_idx];
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
  always @(*) begin
    state_next = state;
    case (state)
      S_IDLE:    if (start) state_next = S_LINEAR1;
      S_LINEAR1: if (ff_idx >= D_FF) state_next = S_RELU;
      S_RELU:    state_next = S_LINEAR2;
      S_LINEAR2: if (out_idx >= D_MODEL) state_next = S_DONE;
      S_DONE:    state_next = S_IDLE;
      default:   state_next = S_IDLE;
    endcase
  end

endmodule
