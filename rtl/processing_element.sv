// =============================================================================
// processing_element.sv - Systolic Array Processing Element
// =============================================================================
// Single MAC (Multiply-Accumulate) unit. Forms the basic compute tile of the
// systolic array used for matrix multiplication in attention and FFN layers.
//
// Data flows: activations flow left-to-right, weights flow top-to-bottom.
// Each PE computes: acc += a_in * w_in, then forwards both operands.
// =============================================================================

module processing_element
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   clear,       // Clear accumulator
  input  logic   enable,      // Enable computation

  // Systolic data flow
  input  signed [15:0]  a_in,        // Activation input (from left)
  input  signed [15:0]  w_in,        // Weight input (from top)
  output reg signed [15:0]  a_out,       // Activation output (to right)
  output reg signed [15:0]  w_out,       // Weight output (to bottom)

  // Result
  output reg signed [31:0]   acc_out      // Accumulated result
);

  reg signed [31:0] accumulator;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out       <= 0;
      w_out       <= 0;
      accumulator <= 0;
    end else if (clear) begin
      a_out       <= 0;
      w_out       <= 0;
      accumulator <= 0;
    end else if (enable) begin
      // Forward operands through systolic array
      a_out <= a_in;
      w_out <= w_in;

      // MAC operation: acc += a * w (full precision multiply)
      accumulator <= accumulator + (to_acc(a_in) * to_acc(w_in));
    end
  end

  assign acc_out = accumulator;

endmodule
