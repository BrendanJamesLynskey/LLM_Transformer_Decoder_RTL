// =============================================================================
// transformer_decoder_stream.sv - Streaming Transformer Decoder Block
// =============================================================================
// Streaming variant of transformer_decoder. Instead of receiving full weight
// arrays on combinational ports, this module exposes BRAM read address/data
// interfaces. The top-level connects these directly to weight BRAMs,
// eliminating the ~49K register array bridge.
//
// Interface summary:
//   - LayerNorm gamma/beta: still packed 1D ports (only 64 elements each,
//     small enough that registers are acceptable)
//   - Attention weights Wq/Wk/Wv/Wo: BRAM address/data interfaces
//   - FFN weights W1/W2/b1: BRAM address/data interfaces
//   - FFN b2: packed 1D port (64 elements)
//   - KV-cache: element-level BRAM read interfaces + vector write
// =============================================================================

module transformer_decoder_stream
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  // Input token embedding (packed 1D)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] token_emb,

  // LayerNorm parameters (packed 1D — small, ok as register ports)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_gamma,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_beta,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_gamma,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_beta,

  // Attention weight BRAM read interfaces
  output logic [$clog2(D_MODEL*D_MODEL)-1:0] wqkv_rd_addr,
  input  data_t  wq_rd_data,
  input  data_t  wk_rd_data,
  input  data_t  wv_rd_data,
  output logic   wqkv_rd_en,

  output logic [$clog2(D_MODEL*D_MODEL)-1:0] wo_rd_addr,
  input  data_t  wo_rd_data,
  output logic   wo_rd_en,

  // FFN weight BRAM read interfaces
  output logic [$clog2(D_MODEL*D_FF)-1:0] w1_rd_addr,
  input  data_t  w1_rd_data,
  output logic   w1_rd_en,

  output logic [$clog2(D_FF)-1:0] b1_rd_addr,
  input  data_t  b1_rd_data,
  output logic   b1_rd_en,

  output logic [$clog2(D_FF*D_MODEL)-1:0] w2_rd_addr,
  input  data_t  w2_rd_data,
  output logic   w2_rd_en,

  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ffn_b2,

  // Sequence position
  input  seq_idx_t seq_pos,

  // KV-Cache write interface
  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr,
  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr,
  output logic   cache_wr_en,

  // KV-Cache BRAM read interfaces
  output seq_idx_t              kcache_rd_pos,
  output logic [$clog2(D_MODEL)-1:0] kcache_rd_dim,
  input  data_t                 kcache_rd_data,
  output seq_idx_t              vcache_rd_pos,
  output logic [$clog2(D_MODEL)-1:0] vcache_rd_dim,
  input  data_t                 vcache_rd_data,

  // Output (packed 1D)
  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] out_emb,
  output logic   valid
);

  // =========================================================================
  // Top-Level FSM
  // =========================================================================
  typedef enum logic [3:0] {
    S_IDLE,
    S_LN1_START,
    S_LN1_WAIT,
    S_ATTN_START,
    S_ATTN_WAIT,
    S_RESIDUAL1,
    S_LN2_START,
    S_LN2_WAIT,
    S_FFN_START,
    S_FFN_WAIT,
    S_RESIDUAL2,
    S_DONE
  } top_state_t;

  top_state_t state, state_next;

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

  // LayerNorm 1 (pre-attention) — unchanged, uses packed ports
  layer_norm #(.VEC_LEN(D_MODEL)) u_ln1 (
    .clk(clk), .rst_n(rst_n), .start(ln1_start),
    .x_in(token_emb), .gamma(ln1_gamma), .beta(ln1_beta),
    .y_out(ln1_out), .valid(ln1_valid)
  );

  // Multi-Head Attention — streaming
  multi_head_attention_stream u_mha (
    .clk(clk), .rst_n(rst_n), .start(attn_start),
    .x_in(ln1_out),
    // Weight BRAM interfaces (directly forwarded to top-level)
    .wqkv_rd_addr(wqkv_rd_addr), .wq_rd_data(wq_rd_data),
    .wk_rd_data(wk_rd_data), .wv_rd_data(wv_rd_data),
    .wqkv_rd_en(wqkv_rd_en),
    .wo_rd_addr(wo_rd_addr), .wo_rd_data(wo_rd_data), .wo_rd_en(wo_rd_en),
    // KV-cache
    .seq_pos(seq_pos),
    .k_cache_wr(k_cache_wr), .v_cache_wr(v_cache_wr), .cache_wr_en(cache_wr_en),
    .kcache_rd_pos(kcache_rd_pos), .kcache_rd_dim(kcache_rd_dim),
    .kcache_rd_data(kcache_rd_data),
    .vcache_rd_pos(vcache_rd_pos), .vcache_rd_dim(vcache_rd_dim),
    .vcache_rd_data(vcache_rd_data),
    .attn_out(attn_out), .valid(attn_valid)
  );

  // LayerNorm 2 (pre-FFN) — unchanged
  layer_norm #(.VEC_LEN(D_MODEL)) u_ln2 (
    .clk(clk), .rst_n(rst_n), .start(ln2_start),
    .x_in(residual1), .gamma(ln2_gamma), .beta(ln2_beta),
    .y_out(ln2_out), .valid(ln2_valid)
  );

  // Feed-Forward Network — streaming
  feed_forward_stream u_ffn (
    .clk(clk), .rst_n(rst_n), .start(ffn_start),
    .x_in(ln2_out),
    .w1_rd_addr(w1_rd_addr), .w1_rd_data(w1_rd_data), .w1_rd_en(w1_rd_en),
    .b1_rd_addr(b1_rd_addr), .b1_rd_data(b1_rd_data), .b1_rd_en(b1_rd_en),
    .w2_rd_addr(w2_rd_addr), .w2_rd_data(w2_rd_data), .w2_rd_en(w2_rd_en),
    .b2(ffn_b2),
    .y_out(ffn_out), .valid(ffn_valid)
  );

  // =========================================================================
  // Top-Level FSM — identical sequencing to original
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      valid      <= 1'b0;
      ln1_start  <= 1'b0;
      attn_start <= 1'b0;
      ln2_start  <= 1'b0;
      ffn_start  <= 1'b0;
      for (int i = 0; i < D_MODEL; i++) begin
        residual1[i] <= '0;
        out_emb[i]   <= '0;
      end
    end else begin
      state      <= state_next;
      ln1_start  <= 1'b0;
      attn_start <= 1'b0;
      ln2_start  <= 1'b0;
      ffn_start  <= 1'b0;

      case (state)
        S_IDLE:        valid <= 1'b0;
        S_LN1_START:   ln1_start  <= 1'b1;
        S_LN1_WAIT:    ;
        S_ATTN_START:  attn_start <= 1'b1;
        S_ATTN_WAIT:   ;
        S_RESIDUAL1:   for (int i = 0; i < D_MODEL; i++)
                         residual1[i] <= fp_sat_add(token_emb[i], attn_out[i]);
        S_LN2_START:   ln2_start <= 1'b1;
        S_LN2_WAIT:    ;
        S_FFN_START:   ffn_start <= 1'b1;
        S_FFN_WAIT:    ;
        S_RESIDUAL2: begin
          for (int i = 0; i < D_MODEL; i++)
            out_emb[i] <= fp_sat_add(residual1[i], ffn_out[i]);
          valid <= 1'b1;
        end
        S_DONE:        ;
        default:       ;
      endcase
    end
  end

  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:       if (start) state_next = S_LN1_START;
      S_LN1_START:  state_next = S_LN1_WAIT;
      S_LN1_WAIT:   if (ln1_valid)  state_next = S_ATTN_START;
      S_ATTN_START: state_next = S_ATTN_WAIT;
      S_ATTN_WAIT:  if (attn_valid) state_next = S_RESIDUAL1;
      S_RESIDUAL1:  state_next = S_LN2_START;
      S_LN2_START:  state_next = S_LN2_WAIT;
      S_LN2_WAIT:   if (ln2_valid)  state_next = S_FFN_START;
      S_FFN_START:  state_next = S_FFN_WAIT;
      S_FFN_WAIT:   if (ffn_valid)  state_next = S_RESIDUAL2;
      S_RESIDUAL2:  state_next = S_DONE;
      S_DONE:       state_next = S_IDLE;
      default:      state_next = S_IDLE;
    endcase
  end

endmodule
