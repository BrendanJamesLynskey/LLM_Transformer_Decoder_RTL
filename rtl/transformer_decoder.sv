// =============================================================================
// transformer_decoder.sv - Single Transformer Decoder Block (Top-Level)
// =============================================================================
// Implements one complete transformer decoder layer for autoregressive
// inference. Composes the sub-modules into the standard architecture:
//
//   residual_1 = x + MultiHeadAttention(LayerNorm(x))
//   residual_2 = residual_1 + FFN(LayerNorm(residual_1))
//   output = residual_2
//
// This is the "pre-norm" architecture used in modern LLMs (GPT-2+, LLaMA, etc.)
//
// For multi-layer stacking, instantiate N copies with shared or separate
// weight memories. The top-level controller sequences token processing
// through all layers.
// =============================================================================

module transformer_decoder
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  // Input token embedding (packed 1D)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] token_emb,

  // Attention weights (unpacked 2D)
  input  signed [15:0]  wq [D_MODEL][D_MODEL],
  input  signed [15:0]  wk [D_MODEL][D_MODEL],
  input  signed [15:0]  wv [D_MODEL][D_MODEL],
  input  signed [15:0]  wo [D_MODEL][D_MODEL],

  // LayerNorm parameters (packed 1D)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_gamma,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_beta,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_gamma,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_beta,

  // FFN weights (unpacked 2D, packed 1D for biases)
  input  signed [15:0]  ffn_w1 [D_MODEL][D_FF],
  input  signed [15:0]  ffn_b1 [D_FF],
  input  signed [15:0]  ffn_w2 [D_FF][D_MODEL],
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ffn_b2,

  // Sequence position
  input  [6:0] seq_pos,

  // KV-Cache
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr,
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr,
  output reg   cache_wr_en,
  input  signed [15:0]  k_cache [MAX_SEQ_LEN][D_MODEL],
  input  signed [15:0]  v_cache [MAX_SEQ_LEN][D_MODEL],

  // Output (packed 1D)
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] out_emb,
  output reg   valid
);

  integer i;
  // =========================================================================
  // Top-Level FSM
  // =========================================================================
  localparam [3:0] S_IDLE = 0;
  localparam [3:0] S_LN1_START = 1;
  localparam [3:0] S_LN1_WAIT = 2;
  localparam [3:0] S_ATTN_START = 3;
  localparam [3:0] S_ATTN_WAIT = 4;
  localparam [3:0] S_RESIDUAL1 = 5;
  localparam [3:0] S_LN2_START = 6;
  localparam [3:0] S_LN2_WAIT = 7;
  localparam [3:0] S_FFN_START = 8;
  localparam [3:0] S_FFN_WAIT = 9;
  localparam [3:0] S_RESIDUAL2 = 10;
  localparam [3:0] S_DONE = 11;

  reg [3:0] state, state_next;

  // Intermediate signals
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_out;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] attn_out;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] residual1;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_out;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ffn_out;

  // Sub-module control
  logic  ln1_start, ln1_valid;
  logic  attn_start, attn_valid;
  logic  ln2_start, ln2_valid;
  logic  ffn_start, ffn_valid;

  // =========================================================================
  // Sub-Module Instantiation
  // =========================================================================

  // LayerNorm 1 (pre-attention)
  layer_norm #(.VEC_LEN(D_MODEL)) u_ln1 (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (ln1_start),
    .x_in   (token_emb),
    .gamma  (ln1_gamma),
    .beta   (ln1_beta),
    .y_out  (ln1_out),
    .valid  (ln1_valid)
  );

  // Multi-Head Attention
  multi_head_attention u_mha (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (attn_start),
    .x_in         (ln1_out),
    .wq           (wq),
    .wk           (wk),
    .wv           (wv),
    .wo           (wo),
    .seq_pos      (seq_pos),
    .k_cache_wr   (k_cache_wr),
    .v_cache_wr   (v_cache_wr),
    .cache_wr_en  (cache_wr_en),
    .k_cache      (k_cache),
    .v_cache      (v_cache),
    .attn_out     (attn_out),
    .valid        (attn_valid)
  );

  // LayerNorm 2 (pre-FFN)
  layer_norm #(.VEC_LEN(D_MODEL)) u_ln2 (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (ln2_start),
    .x_in   (residual1),
    .gamma  (ln2_gamma),
    .beta   (ln2_beta),
    .y_out  (ln2_out),
    .valid  (ln2_valid)
  );

  // Feed-Forward Network
  feed_forward u_ffn (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (ffn_start),
    .x_in   (ln2_out),
    .w1     (ffn_w1),
    .b1     (ffn_b1),
    .w2     (ffn_w2),
    .b2     (ffn_b2),
    .y_out  (ffn_out),
    .valid  (ffn_valid)
  );

  // =========================================================================
  // Top-Level FSM
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      valid      <= 1'b0;
      ln1_start  <= 1'b0;
      attn_start <= 1'b0;
      ln2_start  <= 1'b0;
      ffn_start  <= 1'b0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        residual1[i] <= 0;
        out_emb[i]   <= 0;
      end
    end else begin
      state      <= state_next;
      ln1_start  <= 1'b0;
      attn_start <= 1'b0;
      ln2_start  <= 1'b0;
      ffn_start  <= 1'b0;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
        end

        S_LN1_START: begin
          ln1_start <= 1'b1;
        end

        S_LN1_WAIT: ; // Wait for LN1

        S_ATTN_START: begin
          attn_start <= 1'b1;
        end

        S_ATTN_WAIT: ; // Wait for attention

        // Residual connection 1: residual1 = token_emb + attn_out
        S_RESIDUAL1: begin
          for (i = 0; i < D_MODEL; i = i + 1)
            residual1[i] <= fp_sat_add(token_emb[i], attn_out[i]);
        end

        S_LN2_START: begin
          ln2_start <= 1'b1;
        end

        S_LN2_WAIT: ; // Wait for LN2

        S_FFN_START: begin
          ffn_start <= 1'b1;
        end

        S_FFN_WAIT: ; // Wait for FFN

        // Residual connection 2: out = residual1 + ffn_out
        S_RESIDUAL2: begin
          for (i = 0; i < D_MODEL; i = i + 1)
            out_emb[i] <= fp_sat_add(residual1[i], ffn_out[i]);
          valid <= 1'b1;
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
      S_IDLE:       if (start) state_next = S_LN1_START;
      S_LN1_START:  state_next = S_LN1_WAIT;
      S_LN1_WAIT:   if (ln1_valid)  state_next = S_ATTN_START;
      S_ATTN_START:  state_next = S_ATTN_WAIT;
      S_ATTN_WAIT:   if (attn_valid) state_next = S_RESIDUAL1;
      S_RESIDUAL1:   state_next = S_LN2_START;
      S_LN2_START:   state_next = S_LN2_WAIT;
      S_LN2_WAIT:    if (ln2_valid)  state_next = S_FFN_START;
      S_FFN_START:   state_next = S_FFN_WAIT;
      S_FFN_WAIT:    if (ffn_valid)  state_next = S_RESIDUAL2;
      S_RESIDUAL2:   state_next = S_DONE;
      S_DONE:        state_next = S_IDLE;
      default:       state_next = S_IDLE;
    endcase
  end

endmodule
