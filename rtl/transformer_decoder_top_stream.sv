// =============================================================================
// transformer_decoder_top_stream.sv - Streaming BRAM-Backed Top Level
// =============================================================================
// Streaming variant of transformer_decoder_top that eliminates the ~49K
// register array bridge. Weight BRAMs are connected directly to the
// decoder's BRAM read address/data interfaces.
//
// Architecture:
//   - 6 weight BRAMs (Wq, Wk, Wv, Wo, W1, W2) with shared read ports
//     muxed between the weight-loading bus and compute read addresses
//   - 1 bias BRAM (b1) with shared read port
//   - 4 small register arrays for LN gamma/beta (64 elements each = 256 regs)
//   - 1 packed register for b2 (64 elements = 64 regs)
//   - 2 KV-cache dual-port BRAMs with element-level read
//
// Register savings vs original:
//   Original: 49,344 × 16-bit = 789,504 flip-flops
//   Streaming: 320 × 16-bit = 5,120 flip-flops (LN params + b2 only)
//   Reduction: ~99.4%
//
// Weight-loading bus address map is unchanged from the original.
// =============================================================================

module transformer_decoder_top_stream
  import transformer_pkg::*;
#(
  parameter WQ_INIT   = "",
  parameter WK_INIT   = "",
  parameter WV_INIT   = "",
  parameter WO_INIT   = "",
  parameter W1_INIT   = "",
  parameter W2_INIT   = "",
  parameter LN1G_INIT = "",
  parameter LN1B_INIT = "",
  parameter LN2G_INIT = "",
  parameter LN2B_INIT = "",
  parameter B1_INIT   = "",
  parameter B2_INIT   = ""
)(
  input  logic     clk,
  input  logic     rst_n,

  // --- Weight-loading bus ---
  input  logic     wl_en,
  input  logic [15:0] wl_addr,
  input  data_t    wl_data,

  // --- Inference ---
  input  logic     start,
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] token_emb,
  input  seq_idx_t seq_pos,

  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] out_emb,
  output logic     valid
);

  // =========================================================================
  // Address Map (unchanged from original)
  // =========================================================================
  localparam logic [15:0] BASE_WQ   = 16'h0000;
  localparam logic [15:0] BASE_WK   = 16'h1000;
  localparam logic [15:0] BASE_WV   = 16'h2000;
  localparam logic [15:0] BASE_WO   = 16'h3000;
  localparam logic [15:0] BASE_W1   = 16'h4000;
  localparam logic [15:0] BASE_W2   = 16'h8000;
  localparam logic [15:0] BASE_LN1G = 16'hC000;
  localparam logic [15:0] BASE_LN1B = 16'hC040;
  localparam logic [15:0] BASE_LN2G = 16'hC080;
  localparam logic [15:0] BASE_LN2B = 16'hC0C0;
  localparam logic [15:0] BASE_B1   = 16'hC100;
  localparam logic [15:0] BASE_B2   = 16'hC200;

  // =========================================================================
  // Write-Enable Decode
  // =========================================================================
  logic we_wq, we_wk, we_wv, we_wo, we_w1, we_w2;
  logic we_ln1g, we_ln1b, we_ln2g, we_ln2b, we_b1, we_b2;

  always_comb begin
    we_wq   = wl_en && (wl_addr >= BASE_WQ)   && (wl_addr < BASE_WK);
    we_wk   = wl_en && (wl_addr >= BASE_WK)   && (wl_addr < BASE_WV);
    we_wv   = wl_en && (wl_addr >= BASE_WV)   && (wl_addr < BASE_WO);
    we_wo   = wl_en && (wl_addr >= BASE_WO)   && (wl_addr < BASE_W1);
    we_w1   = wl_en && (wl_addr >= BASE_W1)   && (wl_addr < BASE_W2);
    we_w2   = wl_en && (wl_addr >= BASE_W2)   && (wl_addr < BASE_LN1G);
    we_ln1g = wl_en && (wl_addr >= BASE_LN1G) && (wl_addr < BASE_LN1B);
    we_ln1b = wl_en && (wl_addr >= BASE_LN1B) && (wl_addr < BASE_LN2G);
    we_ln2g = wl_en && (wl_addr >= BASE_LN2G) && (wl_addr < BASE_LN2B);
    we_ln2b = wl_en && (wl_addr >= BASE_LN2B) && (wl_addr < BASE_B1);
    we_b1   = wl_en && (wl_addr >= BASE_B1)   && (wl_addr < BASE_B2);
    we_b2   = wl_en && (wl_addr >= BASE_B2)   && (wl_addr < 16'hC240);
  end

  // =========================================================================
  // Small Register Arrays (LN params + b2 only — 320 registers total)
  // =========================================================================
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_gamma;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_beta;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_gamma;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_beta;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2_arr;

  always_ff @(posedge clk) begin
    if (we_ln1g) ln1_gamma[wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln1b) ln1_beta [wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln2g) ln2_gamma[wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln2b) ln2_beta [wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_b2)   b2_arr   [wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
  end

  // =========================================================================
  // Decoder BRAM Read Signals
  // =========================================================================
  logic [$clog2(D_MODEL*D_MODEL)-1:0] wqkv_rd_addr;
  logic                                wqkv_rd_en;
  data_t wq_rd_data, wk_rd_data, wv_rd_data;

  logic [$clog2(D_MODEL*D_MODEL)-1:0] wo_rd_addr;
  logic                                wo_rd_en;
  data_t wo_rd_data;

  logic [$clog2(D_MODEL*D_FF)-1:0]    w1_rd_addr;
  logic                                w1_rd_en;
  data_t w1_rd_data;

  logic [$clog2(D_FF)-1:0]            b1_rd_addr;
  logic                                b1_rd_en;
  data_t b1_rd_data;

  logic [$clog2(D_FF*D_MODEL)-1:0]    w2_rd_addr;
  logic                                w2_rd_en;
  data_t w2_rd_data;

  // KV-cache signals
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr_vec;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr_vec;
  logic cache_wr_en;
  seq_idx_t kcache_rd_pos, vcache_rd_pos;
  logic [$clog2(D_MODEL)-1:0] kcache_rd_dim, vcache_rd_dim;
  data_t kcache_rd_data, vcache_rd_data;

  // =========================================================================
  // Weight BRAMs — read port muxed between weight-load and compute
  // =========================================================================
  // Wq
  logic [$clog2(D_MODEL*D_MODEL)-1:0] wq_addr_mux;
  assign wq_addr_mux = we_wq ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0]
                              : wqkv_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WQ_INIT))
  u_wq (.clk(clk), .we(we_wq), .addr(wq_addr_mux), .wdata(wl_data), .rdata(wq_rd_data));

  // Wk
  logic [$clog2(D_MODEL*D_MODEL)-1:0] wk_addr_mux;
  assign wk_addr_mux = we_wk ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0]
                              : wqkv_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WK_INIT))
  u_wk (.clk(clk), .we(we_wk), .addr(wk_addr_mux), .wdata(wl_data), .rdata(wk_rd_data));

  // Wv
  logic [$clog2(D_MODEL*D_MODEL)-1:0] wv_addr_mux;
  assign wv_addr_mux = we_wv ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0]
                              : wqkv_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WV_INIT))
  u_wv (.clk(clk), .we(we_wv), .addr(wv_addr_mux), .wdata(wl_data), .rdata(wv_rd_data));

  // Wo
  logic [$clog2(D_MODEL*D_MODEL)-1:0] wo_addr_mux;
  assign wo_addr_mux = we_wo ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0]
                              : wo_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WO_INIT))
  u_wo (.clk(clk), .we(we_wo), .addr(wo_addr_mux), .wdata(wl_data), .rdata(wo_rd_data));

  // W1
  logic [$clog2(D_MODEL*D_FF)-1:0] w1_addr_mux;
  assign w1_addr_mux = we_w1 ? wl_addr[$clog2(D_MODEL*D_FF)-1:0]
                              : w1_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_FF), .INIT_FILE(W1_INIT))
  u_w1 (.clk(clk), .we(we_w1), .addr(w1_addr_mux), .wdata(wl_data), .rdata(w1_rd_data));

  // W2
  logic [$clog2(D_FF*D_MODEL)-1:0] w2_addr_mux;
  assign w2_addr_mux = we_w2 ? (wl_addr - BASE_W2) : w2_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_FF*D_MODEL), .INIT_FILE(W2_INIT))
  u_w2 (.clk(clk), .we(we_w2), .addr(w2_addr_mux), .wdata(wl_data), .rdata(w2_rd_data));

  // b1
  logic [$clog2(D_FF)-1:0] b1_addr_mux;
  assign b1_addr_mux = we_b1 ? wl_addr[$clog2(D_FF)-1:0]
                              : b1_rd_addr;
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_FF), .INIT_FILE(B1_INIT))
  u_b1 (.clk(clk), .we(we_b1), .addr(b1_addr_mux), .wdata(wl_data), .rdata(b1_rd_data));

  // LN gamma/beta and b2 BRAMs (for persistent storage / init)
  // These are small enough that the register arrays above ARE the storage.
  // We still instantiate the BRAMs for hex-init but only use write port.
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN1G_INIT))
  u_ln1g (.clk(clk), .we(we_ln1g), .addr(wl_addr[$clog2(D_MODEL)-1:0]), .wdata(wl_data), .rdata());
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN1B_INIT))
  u_ln1b (.clk(clk), .we(we_ln1b), .addr(wl_addr[$clog2(D_MODEL)-1:0]), .wdata(wl_data), .rdata());
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN2G_INIT))
  u_ln2g (.clk(clk), .we(we_ln2g), .addr(wl_addr[$clog2(D_MODEL)-1:0]), .wdata(wl_data), .rdata());
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN2B_INIT))
  u_ln2b (.clk(clk), .we(we_ln2b), .addr(wl_addr[$clog2(D_MODEL)-1:0]), .wdata(wl_data), .rdata());
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(B2_INIT))
  u_b2 (.clk(clk), .we(we_b2), .addr(wl_addr[$clog2(D_MODEL)-1:0]), .wdata(wl_data), .rdata());

  // =========================================================================
  // KV-Cache BRAMs — element-level read for streaming
  // =========================================================================
  // K-cache: write via vector interface, read element-by-element
  kv_cache_bram u_k_cache (
    .clk(clk), .rst_n(rst_n),
    .wr_start(cache_wr_en), .wr_seq_pos(seq_pos), .wr_vec(k_cache_wr_vec),
    .wr_done(),
    .rd_pos(kcache_rd_pos), .rd_dim(kcache_rd_dim), .rd_data(kcache_rd_data)
  );

  // V-cache: same pattern
  kv_cache_bram u_v_cache (
    .clk(clk), .rst_n(rst_n),
    .wr_start(cache_wr_en), .wr_seq_pos(seq_pos), .wr_vec(v_cache_wr_vec),
    .wr_done(),
    .rd_pos(vcache_rd_pos), .rd_dim(vcache_rd_dim), .rd_data(vcache_rd_data)
  );

  // =========================================================================
  // Streaming Decoder Core
  // =========================================================================
  transformer_decoder_stream u_decoder (
    .clk(clk), .rst_n(rst_n), .start(start),
    .token_emb(token_emb),
    .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
    .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
    // Weight BRAM read interfaces
    .wqkv_rd_addr(wqkv_rd_addr), .wq_rd_data(wq_rd_data),
    .wk_rd_data(wk_rd_data), .wv_rd_data(wv_rd_data),
    .wqkv_rd_en(wqkv_rd_en),
    .wo_rd_addr(wo_rd_addr), .wo_rd_data(wo_rd_data), .wo_rd_en(wo_rd_en),
    .w1_rd_addr(w1_rd_addr), .w1_rd_data(w1_rd_data), .w1_rd_en(w1_rd_en),
    .b1_rd_addr(b1_rd_addr), .b1_rd_data(b1_rd_data), .b1_rd_en(b1_rd_en),
    .w2_rd_addr(w2_rd_addr), .w2_rd_data(w2_rd_data), .w2_rd_en(w2_rd_en),
    .ffn_b2(b2_arr),
    .seq_pos(seq_pos),
    .k_cache_wr(k_cache_wr_vec), .v_cache_wr(v_cache_wr_vec),
    .cache_wr_en(cache_wr_en),
    .kcache_rd_pos(kcache_rd_pos), .kcache_rd_dim(kcache_rd_dim),
    .kcache_rd_data(kcache_rd_data),
    .vcache_rd_pos(vcache_rd_pos), .vcache_rd_dim(vcache_rd_dim),
    .vcache_rd_data(vcache_rd_data),
    .out_emb(out_emb), .valid(valid)
  );

endmodule
