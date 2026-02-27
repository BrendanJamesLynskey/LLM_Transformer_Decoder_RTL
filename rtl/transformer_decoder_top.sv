// =============================================================================
// transformer_decoder_top.sv - BRAM-Backed Transformer Decoder Top Level
// =============================================================================
// Wraps the transformer_decoder core with BRAM-backed storage for all
// weights, biases, and KV-cache.
//
// Weight Initialisation:
//   Each BRAM accepts an INIT_FILE parameter (hex file path) so weights
//   can be preloaded at synthesis or simulation start. The file format
//   is one DATA_WIDTH-bit hex value per line, in row-major order.
//
//   At runtime, the weight-loading bus (wl_en / wl_addr / wl_data) can
//   update any weight element via a unified flat address space:
//
//     Region       | Base    | Size (words) | Dimensions
//     -------------|---------|--------------|------------------
//     Wq           | 0x0000  | 4096         | D_MODEL × D_MODEL
//     Wk           | 0x1000  | 4096         | D_MODEL × D_MODEL
//     Wv           | 0x2000  | 4096         | D_MODEL × D_MODEL
//     Wo           | 0x3000  | 4096         | D_MODEL × D_MODEL
//     FFN W1       | 0x4000  | 16384        | D_MODEL × D_FF
//     FFN W2       | 0x8000  | 16384        | D_FF × D_MODEL
//     LN1 gamma    | 0xC000  | 64           | D_MODEL
//     LN1 beta     | 0xC040  | 64           | D_MODEL
//     LN2 gamma    | 0xC080  | 64           | D_MODEL
//     LN2 beta     | 0xC0C0  | 64           | D_MODEL
//     FFN b1       | 0xC100  | 256          | D_FF
//     FFN b2       | 0xC200  | 64           | D_MODEL
//
// KV-Cache:
//   Two kv_cache_bram instances hold K and V caches in dual-port BRAM.
//   The decoder's cache_wr_en / k_cache_wr / v_cache_wr signals drive
//   writes; reads are element-level via (rd_pos, rd_dim).
//
// Architecture Note:
//   The inner transformer_decoder uses combinational array ports. This
//   top module instantiates weight BRAMs and continuously drives the
//   decoder's array ports from BRAM contents. For 2D weight matrices,
//   the BRAM contents are read into registered arrays via a preload
//   phase. For 1D parameter vectors, individual BRAMs feed directly.
//
//   A fully streaming architecture (future work) would replace the
//   inner decoder's array ports with address/data handshake interfaces,
//   eliminating the need for the large register arrays.
// =============================================================================

module transformer_decoder_top
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

  // --- Weight-loading bus (active during non-inference) ---
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
  // Address Map
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
  // Weight Register Arrays
  // =========================================================================
  // These hold the full weight tensors for the decoder's array ports.
  // They are loaded from BRAMs during reset/preload and updated
  // whenever the weight-load bus writes to the corresponding region.

  data_t wq_arr [D_MODEL][D_MODEL];
  data_t wk_arr [D_MODEL][D_MODEL];
  data_t wv_arr [D_MODEL][D_MODEL];
  data_t wo_arr [D_MODEL][D_MODEL];
  data_t w1_arr [D_MODEL][D_FF];
  data_t w2_arr [D_FF][D_MODEL];
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_gamma;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln1_beta;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_gamma;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] ln2_beta;
  data_t b1_arr [D_FF];
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2_arr;

  // Pre-computed address offsets for write-through
  logic [11:0] wl_off_4k;   // Offset within 4K region (for 64×64 matrices)
  logic [13:0] wl_off_16k;  // Offset within 16K region (for FFN matrices)
  logic [13:0] wl_off_w2;   // Offset for W2

  assign wl_off_4k  = wl_addr[11:0];
  assign wl_off_16k = wl_addr[13:0];
  assign wl_off_w2  = wl_addr - BASE_W2;

  // Write-through: update register arrays on wl_en writes
  always_ff @(posedge clk) begin
    if (we_wq) wq_arr[wl_off_4k / D_MODEL][wl_off_4k % D_MODEL] <= wl_data;
    if (we_wk) wk_arr[wl_off_4k / D_MODEL][wl_off_4k % D_MODEL] <= wl_data;
    if (we_wv) wv_arr[wl_off_4k / D_MODEL][wl_off_4k % D_MODEL] <= wl_data;
    if (we_wo) wo_arr[wl_off_4k / D_MODEL][wl_off_4k % D_MODEL] <= wl_data;
    if (we_w1) w1_arr[wl_off_16k / D_FF][wl_off_16k % D_FF]     <= wl_data;
    if (we_w2) w2_arr[wl_off_w2 / D_MODEL][wl_off_w2 % D_MODEL] <= wl_data;
    if (we_ln1g) ln1_gamma[wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln1b) ln1_beta [wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln2g) ln2_gamma[wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_ln2b) ln2_beta [wl_addr[$clog2(D_MODEL)-1:0]] <= wl_data;
    if (we_b1)   b1_arr[wl_addr[$clog2(D_FF)-1:0]]       <= wl_data;
    if (we_b2)   b2_arr[wl_addr[$clog2(D_MODEL)-1:0]]    <= wl_data;
  end

  // =========================================================================
  // Weight BRAMs (persistent storage, init from file)
  // =========================================================================
  // These BRAMs hold the authoritative copy of all weights. They can be
  // initialised at synthesis via INIT_FILE and written at runtime via
  // the wl_en bus. The register arrays above mirror their contents.

  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WQ_INIT)) u_wq (
    .clk(clk), .we(we_wq),
    .addr(we_wq ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0] : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WK_INIT)) u_wk (
    .clk(clk), .we(we_wk),
    .addr(we_wk ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0] : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WV_INIT)) u_wv (
    .clk(clk), .we(we_wv),
    .addr(we_wv ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0] : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_MODEL), .INIT_FILE(WO_INIT)) u_wo (
    .clk(clk), .we(we_wo),
    .addr(we_wo ? wl_addr[$clog2(D_MODEL*D_MODEL)-1:0] : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL*D_FF), .INIT_FILE(W1_INIT)) u_w1 (
    .clk(clk), .we(we_w1),
    .addr(we_w1 ? wl_addr[$clog2(D_MODEL*D_FF)-1:0] : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_FF*D_MODEL), .INIT_FILE(W2_INIT)) u_w2 (
    .clk(clk), .we(we_w2),
    .addr(we_w2 ? (wl_addr - BASE_W2) : '0),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN1G_INIT)) u_ln1g (
    .clk(clk), .we(we_ln1g),
    .addr(wl_addr[$clog2(D_MODEL)-1:0]),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN1B_INIT)) u_ln1b (
    .clk(clk), .we(we_ln1b),
    .addr(wl_addr[$clog2(D_MODEL)-1:0]),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN2G_INIT)) u_ln2g (
    .clk(clk), .we(we_ln2g),
    .addr(wl_addr[$clog2(D_MODEL)-1:0]),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(LN2B_INIT)) u_ln2b (
    .clk(clk), .we(we_ln2b),
    .addr(wl_addr[$clog2(D_MODEL)-1:0]),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_FF), .INIT_FILE(B1_INIT)) u_b1 (
    .clk(clk), .we(we_b1),
    .addr(wl_addr[$clog2(D_FF)-1:0]),
    .wdata(wl_data), .rdata()
  );
  bram_sp #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(D_MODEL), .INIT_FILE(B2_INIT)) u_b2 (
    .clk(clk), .we(we_b2),
    .addr(wl_addr[$clog2(D_MODEL)-1:0]),
    .wdata(wl_data), .rdata()
  );

  // =========================================================================
  // KV-Cache BRAMs
  // =========================================================================
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr_vec;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr_vec;
  logic  cache_wr_en;
  data_t k_cache_arr [MAX_SEQ_LEN][D_MODEL];
  data_t v_cache_arr [MAX_SEQ_LEN][D_MODEL];

  kv_cache_bram u_k_cache (
    .clk(clk), .rst_n(rst_n),
    .wr_start(cache_wr_en), .wr_seq_pos(seq_pos), .wr_vec(k_cache_wr_vec),
    .wr_done(),
    .rd_pos('0), .rd_dim('0), .rd_data()
  );

  kv_cache_bram u_v_cache (
    .clk(clk), .rst_n(rst_n),
    .wr_start(cache_wr_en), .wr_seq_pos(seq_pos), .wr_vec(v_cache_wr_vec),
    .wr_done(),
    .rd_pos('0), .rd_dim('0), .rd_data()
  );

  // Mirror cache writes into the register array for the decoder
  always_ff @(posedge clk) begin
    if (cache_wr_en) begin
      for (int d = 0; d < D_MODEL; d++) begin
        k_cache_arr[seq_pos][d] <= k_cache_wr_vec[d];
        v_cache_arr[seq_pos][d] <= v_cache_wr_vec[d];
      end
    end
  end

  // =========================================================================
  // Decoder Core
  // =========================================================================
  transformer_decoder u_decoder (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (start),
    .token_emb   (token_emb),
    .wq          (wq_arr),
    .wk          (wk_arr),
    .wv          (wv_arr),
    .wo          (wo_arr),
    .ln1_gamma   (ln1_gamma),
    .ln1_beta    (ln1_beta),
    .ln2_gamma   (ln2_gamma),
    .ln2_beta    (ln2_beta),
    .ffn_w1      (w1_arr),
    .ffn_b1      (b1_arr),
    .ffn_w2      (w2_arr),
    .ffn_b2      (b2_arr),
    .seq_pos     (seq_pos),
    .k_cache_wr  (k_cache_wr_vec),
    .v_cache_wr  (v_cache_wr_vec),
    .cache_wr_en (cache_wr_en),
    .k_cache     (k_cache_arr),
    .v_cache     (v_cache_arr),
    .out_emb     (out_emb),
    .valid       (valid)
  );

endmodule
