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

  // Reciprocal square root: 1/sqrt(x) in Q8.8
  // Uses CLZ normalisation + 32-entry LUT + one Newton-Raphson iteration.
  // Accuracy: ~12 bits after N-R refinement, sufficient for Q8.8.
  // Same architectural pattern as softmax compute_reciprocal.
  //
  // Algorithm:
  //   1. CLZ-normalise x to x_norm in [0.5, 1.0) as Q0.16
  //   2. LUT: r0 ≈ 1/√x_norm in Q2.14, indexed by x_norm[14:10]
  //   3. Newton-Raphson: r1 = r0 * (3 - x_norm * r0²) / 2  (all Q2.14)
  //   4. Denormalise: rsqrt = r1 >> (14 - FRAC_BITS - lz/2 adjustment)
  //
  // Since 1/√(x * 2^(-lz)) = 1/√x * 2^(lz/2):
  //   rsqrt(x) = rsqrt(x_norm) * 2^(lz/2)
  //   In Q2.14 → Q8.8: shift right by (14 - FRAC_BITS) = 6, then adjust for lz.
  //   For even lz: multiply by 2^(lz/2)  → net shift = 6 - lz/2
  //   For odd lz: multiply by 2^((lz-1)/2) * √2 → shift = 6 - (lz-1)/2, then ×√2
  //
  //   √2 ≈ 1.4142 in Q2.14 = 23170

  // 32-entry rsqrt LUT: 1/√(0.5 + k/64 + 1/128) in Q2.14
  function automatic logic [15:0] rsqrt_lut(input logic [4:0] index);
    case (index)
      5'd0:  return 16'd22992;  // 1/sqrt(0.5078)
      5'd1:  return 16'd22646;  // 1/sqrt(0.5234)
      5'd2:  return 16'd22315;  // 1/sqrt(0.5391)
      5'd3:  return 16'd21999;  // 1/sqrt(0.5547)
      5'd4:  return 16'd21695;  // 1/sqrt(0.5703)
      5'd5:  return 16'd21404;  // 1/sqrt(0.5859)
      5'd6:  return 16'd21124;  // 1/sqrt(0.6016)
      5'd7:  return 16'd20855;  // 1/sqrt(0.6172)
      5'd8:  return 16'd20596;  // 1/sqrt(0.6328)
      5'd9:  return 16'd20346;  // 1/sqrt(0.6484)
      5'd10: return 16'd20106;  // 1/sqrt(0.6641)
      5'd11: return 16'd19873;  // 1/sqrt(0.6797)
      5'd12: return 16'd19649;  // 1/sqrt(0.6953)
      5'd13: return 16'd19431;  // 1/sqrt(0.7109)
      5'd14: return 16'd19221;  // 1/sqrt(0.7266)
      5'd15: return 16'd19018;  // 1/sqrt(0.7422)
      5'd16: return 16'd18821;  // 1/sqrt(0.7578)
      5'd17: return 16'd18630;  // 1/sqrt(0.7734)
      5'd18: return 16'd18444;  // 1/sqrt(0.7891)
      5'd19: return 16'd18264;  // 1/sqrt(0.8047)
      5'd20: return 16'd18090;  // 1/sqrt(0.8203)
      5'd21: return 16'd17920;  // 1/sqrt(0.8359)
      5'd22: return 16'd17755;  // 1/sqrt(0.8516)
      5'd23: return 16'd17594;  // 1/sqrt(0.8672)
      5'd24: return 16'd17438;  // 1/sqrt(0.8828)
      5'd25: return 16'd17285;  // 1/sqrt(0.8984)
      5'd26: return 16'd17137;  // 1/sqrt(0.9141)
      5'd27: return 16'd16992;  // 1/sqrt(0.9297)
      5'd28: return 16'd16851;  // 1/sqrt(0.9453)
      5'd29: return 16'd16714;  // 1/sqrt(0.9609)
      5'd30: return 16'd16579;  // 1/sqrt(0.9766)
      5'd31: return 16'd16448;  // 1/sqrt(0.9922)
    endcase
  endfunction

  // Count leading zeros for 16-bit unsigned value
  function automatic logic [3:0] clz16(input logic [15:0] val);
    for (int i = 15; i >= 0; i--) begin
      if (val[i]) return 4'(15 - i);
    end
    return 4'd15;
  endfunction

  // Compute 1/sqrt(x) for Q8.8 input, returning Q8.8 result.
  function automatic data_t fp_inv_sqrt(input data_t x);
    logic [15:0] xu;         // Unsigned version of x (variance is always >= 0)
    logic [3:0]  lz;         // Leading zero count
    logic [15:0] x_norm;     // Normalised to [0.5, 1.0) in Q0.16
    logic [4:0]  lut_idx;
    logic [15:0] r0;         // Initial estimate in Q2.14
    logic [31:0] r0_sq;      // r0 * r0
    logic [15:0] r0_sq_16;   // truncated to Q2.14
    logic [31:0] xr2;        // x_norm * r0^2
    logic [15:0] xr2_16;     // truncated
    logic [15:0] three_minus; // 3.0_Q2.14 - xr2_16
    logic [31:0] r1_wide;    // r0 * three_minus
    logic [15:0] r1;         // >> 15 (the /2 from the formula + alignment)
    logic [15:0] result;

    // Handle edge cases
    if (x <= 0) return int_to_fp(1);        // 1/sqrt(0) -> clamp to 1.0
    if (x == 16'sh0001) return 16'sh1000;   // 1/sqrt(1/256) = 16.0

    xu = x[15:0];

    // Step 1: Count leading zeros and normalise
    lz = clz16(xu);
    x_norm = xu << lz;    // MSB is now bit 15; x_norm in [0.5, 1.0) as Q0.16

    // Step 2: LUT lookup
    lut_idx = x_norm[14:10];
    r0 = rsqrt_lut(lut_idx);  // ≈ 1/√x_norm in Q2.14

    // Step 3: Newton-Raphson iteration
    //   r1 = r0 * (3 - x_norm * r0^2) / 2
    //
    // All arithmetic in Q2.14 domain:
    //   r0^2: Q2.14 * Q2.14 = Q4.28; >> 14 gives Q4.14, but values in [1,2] so Q2.14 ok
    //   x_norm * r0^2: Q0.16 * Q2.14 = 32-bit; >> 16 gives Q2.14
    //   3.0 in Q2.14 = 49152
    //   r0 * (3 - x*r0^2): Q2.14 * Q2.14 = Q4.28; >> 15 gives Q2.14 (the /2)
    r0_sq = {16'b0, r0} * {16'b0, r0};
    r0_sq_16 = r0_sq[29:14];           // >> 14, keep Q2.14
    xr2 = {16'b0, x_norm} * {16'b0, r0_sq_16};
    xr2_16 = xr2[31:16];               // >> 16, gives Q2.14
    three_minus = 16'd49152 - xr2_16;  // 3.0_Q2.14 = 3 * 16384 = 49152
    r1_wide = {16'b0, r0} * {16'b0, three_minus};
    r1 = r1_wide[29:15];               // >> 15 = (>> 14 for Q alignment) then /2

    // Step 4: Denormalise
    // r1 ≈ 1/√x_norm in Q2.14 where x_norm = xu << lz (Q0.16).
    // xu = x in Q8.8 raw bits.  float_val = xu * 2^(-8).
    // x_norm_float = x_norm * 2^(-16) = xu * 2^(lz-16) = float_val * 2^(lz-8).
    //
    // 1/√float_val = 1/√x_norm_float * 2^((lz-8)/2)
    //              = r1 * 2^(-14) * 2^((lz-8)/2)
    //
    // In Q8.8: result = 1/√float_val * 2^8 = r1 * 2^((lz-8)/2 - 6)
    //
    // Let e = lz - 8 (signed, range -8..+7 for typical inputs).
    // If e is even:  result = r1 >> (6 - e/2)
    // If e is odd:   result = (r1 * √2) >> (6 - (e-1)/2)
    //   where √2 in Q2.14 = 23170, and the multiply is done first to
    //   preserve precision before the right-shift.
    //
    // Note: for lz > 12 (very small inputs), 6 - e/2 < 0, meaning left-shift.

    begin
      logic signed [4:0] e;         // lz - 8, range [-8, +7]
      logic        e_odd;
      logic signed [4:0] shift_amt;
      logic [31:0] r1_adj;          // r1 or r1 * √2

      e = {1'b0, lz} - 5'sd8;
      e_odd = e[0];

      if (e_odd) begin
        // Odd e: multiply r1 by √2 first (in full precision), then shift
        r1_adj = {16'b0, r1} * 32'd23170;
        r1_adj = {18'b0, r1_adj[31:14]};    // >> 14, back to Q2.14 range
        shift_amt = 5'sd6 - (e - 5'sd1) / 5'sd2;
      end else begin
        r1_adj = {16'b0, r1};
        shift_amt = 5'sd6 - e / 5'sd2;
      end

      // Apply shift (positive = right-shift, negative = left-shift)
      if (shift_amt >= 0)
        result = r1_adj[15:0] >> shift_amt[3:0];
      else
        result = r1_adj[15:0] << (-shift_amt[3:0]);
    end

    // Clamp to max positive Q8.8
    if (result > 16'h7FFF)
      return 16'sh7FFF;
    else if (result == 0)
      return 16'sh0001;  // Minimum non-zero
    else
      return data_t'(result);
  endfunction

endpackage
