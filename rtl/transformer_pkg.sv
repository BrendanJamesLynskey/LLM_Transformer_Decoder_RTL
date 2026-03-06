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
  localparam int D_MODEL      = 64;    // Model embedding dimension
  localparam int N_HEADS      = 4;     // Number of attention heads
  localparam int D_HEAD       = D_MODEL / N_HEADS; // Per-head dimension (16)
  localparam int D_FF         = 256;   // Feed-forward inner dimension (4x)
  localparam int MAX_SEQ_LEN  = 128;   // Maximum sequence length
  localparam int VOCAB_SIZE   = 512;   // Vocabulary size

  // =========================================================================
  // Fixed-Point Arithmetic Parameters (Q8.8 format)
  // =========================================================================
  localparam int DATA_WIDTH   = 16;    // Total bit width
  localparam int FRAC_BITS    = 8;     // Fractional bits
  localparam int INT_BITS     = DATA_WIDTH - FRAC_BITS; // Integer bits

  // Accumulator uses wider precision to prevent overflow
  localparam int ACC_WIDTH    = 32;    // Accumulator width for MAC operations

  // =========================================================================
  // Hardware Configuration
  // =========================================================================
  localparam int PE_ROWS      = 4;     // Systolic array rows
  localparam int PE_COLS      = 4;     // Systolic array columns

  // =========================================================================
  // Types
  // =========================================================================

  // =========================================================================
  // Packed Array Types (for module ports — iverilog compatibility)
  // =========================================================================
  // iverilog 12.0 cannot propagate values through unpacked array ports.
  // These packed types are used at module boundaries; internal logic may
  // freely use unpacked arrays with pack/unpack helper macros.

  // =========================================================================
  // Type Cast Helpers (iverilog 10.1 compatibility)
  // =========================================================================
  function automatic signed [31:0] to_acc(input signed [15:0] val);
    to_acc = {{16{val[15]}}, val};
  endfunction

  function automatic signed [15:0] to_data(input signed [31:0] val);
    to_data = val[15:0];
  endfunction

  // =========================================================================
  // Fixed-Point Utility Functions
  // =========================================================================

  // Multiply two Q8.8 values, return Q8.8 (truncated)
  function automatic signed [15:0] fp_mul(input signed [15:0] a, input signed [15:0] b);
    logic signed [2*DATA_WIDTH-1:0] product;
    product = a * b;
    return to_data(product >>> FRAC_BITS);
  endfunction

  // Saturating add for Q8.8
  function automatic signed [15:0] fp_sat_add(input signed [15:0] a, input signed [15:0] b);
    logic signed [DATA_WIDTH:0] sum;
    sum = {a[DATA_WIDTH-1], a} + {b[DATA_WIDTH-1], b};
    if (sum > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
      return {1'b0, {(DATA_WIDTH-1){1'b1}}}; // Max positive
    else if (sum < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
      return {1'b1, {(DATA_WIDTH-1){1'b0}}}; // Min negative
    else
      return to_data(sum[DATA_WIDTH-1:0]);
  endfunction

  // Convert integer to Q8.8
  function automatic signed [15:0] int_to_fp(input signed [31:0] val);
    return to_data(val <<< FRAC_BITS);
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
  function automatic [15:0] rsqrt_lut(input logic [4:0] index);
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
  function automatic [3:0] clz16(input logic [15:0] val);
    integer i;
    for (i = 15; i >= 0; i--) begin
      if (val[i]) return (15 - i);
    end
    return 4'd15;
  endfunction

  // Compute 1/sqrt(x) for Q8.8 input, returning Q8.8 result.
  // Note: all logic declarations at function scope (not in nested begin blocks)
  // to avoid iverilog 10.1 runtime crash (vthread_get_rd_context_item assertion).
  function automatic signed [15:0] fp_inv_sqrt(input signed [15:0] x);
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
    logic signed [4:0] e;         // lz - 8, range [-8, +7]
    logic        e_odd;
    logic signed [4:0] shift_amt;
    logic [31:0] r1_adj;          // r1 or r1 * sqrt(2)

    // Handle edge cases
    if (x <= 0) return int_to_fp(1);        // 1/sqrt(0) -> clamp to 1.0
    if (x == 16'sh0001) return 16'sh1000;   // 1/sqrt(1/256) = 16.0

    xu = x[15:0];

    // Step 1: Count leading zeros and normalise
    lz = clz16(xu);
    x_norm = xu << lz;    // MSB is now bit 15; x_norm in [0.5, 1.0) as Q0.16

    // Step 2: LUT lookup
    lut_idx = x_norm[14:10];
    r0 = rsqrt_lut(lut_idx);  // ~= 1/sqrt(x_norm) in Q2.14

    // Step 3: Newton-Raphson iteration
    //   r1 = r0 * (3 - x_norm * r0^2) / 2
    r0_sq = {16'b0, r0} * {16'b0, r0};
    r0_sq_16 = r0_sq[29:14];           // >> 14, keep Q2.14
    xr2 = {16'b0, x_norm} * {16'b0, r0_sq_16};
    xr2_16 = xr2[31:16];               // >> 16, gives Q2.14
    three_minus = 16'd49152 - xr2_16;  // 3.0_Q2.14 = 3 * 16384 = 49152
    r1_wide = {16'b0, r0} * {16'b0, three_minus};
    r1 = r1_wide[29:15];               // >> 15 = (>> 14 for Q alignment) then /2

    // Step 4: Denormalise
    // r1 ~= 1/sqrt(x_norm) in Q2.14 where x_norm = xu << lz (Q0.16).
    // Result in Q8.8: r1 * 2^((lz-8)/2 - 6)
    // For odd e: multiply by sqrt(2) ~= 23170 in Q2.14
    e = {1'b0, lz} - 5'sd8;
    e_odd = e[0];

    if (e_odd) begin
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

    // Clamp to max positive Q8.8
    if (result > 16'h7FFF)
      return 16'sh7FFF;
    else if (result == 0)
      return 16'sh0001;  // Minimum non-zero
    else
      return to_data(result);
  endfunction

endpackage
