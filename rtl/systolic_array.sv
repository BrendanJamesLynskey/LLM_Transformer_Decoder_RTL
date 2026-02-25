// =============================================================================
// systolic_array.sv - Systolic Array for Matrix Multiplication
// =============================================================================
// PE_ROWS x PE_COLS systolic array implementing C = A * B^T.
// Used as the core compute engine for:
//   - Q*K^T attention score computation
//   - Attention * V projection
//   - Feed-forward network linear layers
//
// Operation: Streams A rows from the left and B columns from the top.
// After PE_COLS + PE_ROWS - 1 cycles of streaming, results are available
// in the accumulator outputs.
// =============================================================================

module systolic_array
  import transformer_pkg::*;
#(
  parameter int ROWS = PE_ROWS,
  parameter int COLS = PE_COLS
)(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   clear,
  input  logic   enable,

  // Input interfaces
  input  data_t  a_in  [ROWS],  // Row activations (left edge)
  input  data_t  b_in  [COLS],  // Column weights (top edge)

  // Output: accumulated results from each PE
  output acc_t   result [ROWS][COLS],
  output logic   done
);

  // Internal wires connecting PEs
  data_t a_wire [ROWS][COLS+1];
  data_t w_wire [ROWS+1][COLS];

  // Cycle counter for completion detection
  logic [$clog2(ROWS+COLS):0] cycle_cnt;

  // Connect inputs to left and top edges
  always_comb begin
    for (int r = 0; r < ROWS; r++)
      a_wire[r][0] = a_in[r];
    for (int c = 0; c < COLS; c++)
      w_wire[0][c] = b_in[c];
  end

  // Instantiate PE grid
  genvar r, c;
  generate
    for (r = 0; r < ROWS; r++) begin : gen_row
      for (c = 0; c < COLS; c++) begin : gen_col
        processing_element u_pe (
          .clk     (clk),
          .rst_n   (rst_n),
          .clear   (clear),
          .enable  (enable),
          .a_in    (a_wire[r][c]),
          .w_in    (w_wire[r][c]),
          .a_out   (a_wire[r][c+1]),
          .w_out   (w_wire[r+1][c]),
          .acc_out (result[r][c])
        );
      end
    end
  endgenerate

  // Completion counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cycle_cnt <= '0;
      done      <= 1'b0;
    end else if (clear) begin
      cycle_cnt <= '0;
      done      <= 1'b0;
    end else if (enable && !done) begin
      cycle_cnt <= cycle_cnt + 1;
      if (cycle_cnt == (ROWS + COLS - 1))
        done <= 1'b1;
    end
  end

endmodule
