// =============================================================================
// transformer_pkg.sv - Transformer Decoder Accelerator Parameter Package
// =============================================================================
// Defines all configurable parameters, types, and utility functions for the
// LLM Transformer Decoder inference accelerator.
// =============================================================================

package transformer_pkg;

  // =========================================================================
  // Model Architecture Parameters
  // =========================================================================
  parameter int D_MODEL      = 64;    // Model embedding dimension
  parameter int N_HEADS      = 4;     // Number of attention heads
  parameter int D_HEAD       = D_MODEL / N_HEADS; // Per-head dimension (16)
  parameter int D_FF         = 256;   // Feed-forward inner dimension (4x)
  parameter int MAX_SEQ_LEN  = 128;   // Maximum sequence length
  parameter int VOCAB_SIZE   = 512;   // Vocabulary size

  // =========================================================================
  // Fixed-Point Arithmetic Parameters (Q8.8 format)
  // =========================================================================
  parameter int DATA_WIDTH   = 16;    // Total bit width
  parameter int FRAC_BITS    = 8;     // Fractional bits
  parameter int INT_BITS     = DATA_WIDTH - FRAC_BITS; // Integer bits

  // Accumulator uses wider precision to prevent overflow
  parameter int ACC_WIDTH    = 32;    // Accumulator width for MAC operations

  // =========================================================================
  // Hardware Configuration
  // =========================================================================
  parameter int PE_ROWS      = 4;     // Systolic array rows
  parameter int PE_COLS      = 4;     // Systolic array columns

  // =========================================================================
  // Types
  // =========================================================================
  typedef logic signed [DATA_WIDTH-1:0]  data_t;     // Q8.8 fixed-point
  typedef logic signed [ACC_WIDTH-1:0]   acc_t;      // Accumulator
  typedef logic [$clog2(MAX_SEQ_LEN)-1:0] seq_idx_t; // Sequence index
  typedef logic [$clog2(VOCAB_SIZE)-1:0]  vocab_idx_t;// Vocab index

  // =========================================================================
  // Packed Array Types (for module ports — iverilog compatibility)
  // =========================================================================
  // iverilog 12.0 cannot propagate values through unpacked array ports.
  // These packed types are used at module boundaries; internal logic may
  // freely use unpacked arrays with pack/unpack helper macros.
  typedef logic signed [D_MODEL-1:0][DATA_WIDTH-1:0]     model_vec_t;   // D_MODEL-wide vector
  typedef logic signed [D_FF-1:0][DATA_WIDTH-1:0]        ff_vec_t;      // D_FF-wide vector
  typedef logic signed [D_MODEL-1:0][D_MODEL-1:0][DATA_WIDTH-1:0] model_mat_t; // D_MODEL×D_MODEL matrix
  typedef logic signed [D_MODEL-1:0][D_FF-1:0][DATA_WIDTH-1:0]    model_ff_mat_t; // D_MODEL×D_FF matrix
  typedef logic signed [D_FF-1:0][D_MODEL-1:0][DATA_WIDTH-1:0]    ff_model_mat_t; // D_FF×D_MODEL matrix
  typedef logic signed [MAX_SEQ_LEN-1:0][D_MODEL-1:0][DATA_WIDTH-1:0] cache_t; // Cache matrix

  // =========================================================================
  // Fixed-Point Utility Functions
  // =========================================================================

  // Multiply two Q8.8 values, return Q8.8 (truncated)
  function automatic data_t fp_mul(input data_t a, input data_t b);
    logic signed [2*DATA_WIDTH-1:0] product;
    product = a * b;
    return data_t'(product >>> FRAC_BITS);
  endfunction

  // Saturating add for Q8.8
  function automatic data_t fp_sat_add(input data_t a, input data_t b);
    logic signed [DATA_WIDTH:0] sum;
    sum = {a[DATA_WIDTH-1], a} + {b[DATA_WIDTH-1], b};
    if (sum > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
      return {1'b0, {(DATA_WIDTH-1){1'b1}}}; // Max positive
    else if (sum < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
      return {1'b1, {(DATA_WIDTH-1){1'b0}}}; // Min negative
    else
      return data_t'(sum[DATA_WIDTH-1:0]);
  endfunction

  // Convert integer to Q8.8
  function automatic data_t int_to_fp(input int val);
    return data_t'(val <<< FRAC_BITS);
  endfunction

  // Approximate reciprocal square root (for LayerNorm) via LUT + Newton
  // Returns 1/sqrt(x) in Q8.8 format
  function automatic data_t fp_inv_sqrt(input data_t x);
    // Simplified: return 1.0 for small x, approximate for others
    if (x <= 0) return int_to_fp(1);
    if (x < 16'sh0100) return 16'sh0400; // ~4.0 for very small x
    if (x < 16'sh0400) return 16'sh0200; // ~2.0
    if (x < 16'sh1000) return 16'sh0100; // ~1.0
    return 16'sh0080;                     // ~0.5 for large x
  endfunction

endpackage
