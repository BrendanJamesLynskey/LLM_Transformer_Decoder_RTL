// =============================================================================
// multi_head_attention_stream.sv - Streaming Multi-Head Attention
// =============================================================================
// Streaming variant of multi_head_attention that reads weights from BRAM
// via address/data interfaces instead of combinational array ports.
//
// Weight access pattern:
//   - QKV projection: for each output dim d, read wq[j][d], wk[j][d], wv[j][d]
//     for j = 0..D_MODEL-1. BRAM address = j * D_MODEL + d (row-major).
//   - Output projection: same pattern with wo[j][d].
//   - KV-cache: read k_cache[pos][dim], v_cache[pos][dim] element-by-element.
//
// The FSM generates addresses one cycle before the data is needed to
// account for BRAM's single-cycle read latency. Dot-product accumulation
// is fully serialised: one multiply-accumulate per cycle per projection.
// =============================================================================

module multi_head_attention_stream
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  // Input token embedding (packed 1D)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] x_in,

  // Weight BRAM read interfaces (active during compute)
  // All share the same address since Wq/Wk/Wv are read in lockstep
  output reg [$clog2(D_MODEL*D_MODEL)-1:0] wqkv_rd_addr,
  input  signed [15:0]  wq_rd_data,
  input  signed [15:0]  wk_rd_data,
  input  signed [15:0]  wv_rd_data,
  output reg   wqkv_rd_en,

  output reg [$clog2(D_MODEL*D_MODEL)-1:0] wo_rd_addr,
  input  signed [15:0]  wo_rd_data,
  output reg   wo_rd_en,

  // KV Cache BRAM interfaces
  input  [6:0]              seq_pos,
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr,
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr,
  output reg                  cache_wr_en,

  // KV-cache read interface (element-level)
  output reg [6:0]              kcache_rd_pos,
  output reg [$clog2(D_MODEL)-1:0] kcache_rd_dim,
  input  signed [15:0]                 kcache_rd_data,
  output reg [6:0]              vcache_rd_pos,
  output reg [$clog2(D_MODEL)-1:0] vcache_rd_dim,
  input  signed [15:0]                 vcache_rd_data,

  // Output (packed 1D)
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] attn_out,
  output reg   valid
);

  integer d, h, i, t;
  // =========================================================================
  // State Machine
  // =========================================================================
  localparam [4:0] S_IDLE = 0;
  localparam [4:0] S_PROJ_ADDR = 1;
  localparam [4:0] S_PROJ_QKV = 2;
  localparam [4:0] S_PROJ_STORE = 3;
  localparam [4:0] S_WRITE_CACHE = 4;
  localparam [4:0] S_SCORE_ADDR = 5;
  localparam [4:0] S_SCORE = 6;
  localparam [4:0] S_SCORE_STORE = 7;
  localparam [4:0] S_SOFTMAX_PREP = 8;
  localparam [4:0] S_SOFTMAX_RUN = 9;
  localparam [4:0] S_SOFTMAX_STORE = 10;
  localparam [4:0] S_WSUM_ADDR = 11;
  localparam [4:0] S_WEIGHTED_SUM = 12;
  localparam [4:0] S_WSUM_STORE = 13;
  localparam [4:0] S_CONCAT = 14;
  localparam [4:0] S_OPROJ_ADDR = 15;
  localparam [4:0] S_OUTPUT_PROJ = 16;
  localparam [4:0] S_OPROJ_STORE = 17;
  localparam [4:0] S_DONE = 18;

  reg [4:0] state, state_next;

  // Internal storage
  reg signed [15:0] q_vec [D_MODEL];
  reg signed [15:0] k_vec [D_MODEL];
  reg signed [15:0] v_vec [D_MODEL];
  reg signed [15:0] head_scores [N_HEADS][MAX_SEQ_LEN];
  reg signed [15:0] head_probs  [N_HEADS][MAX_SEQ_LEN];
  reg signed [15:0] head_out    [N_HEADS][D_HEAD];
  reg signed [15:0] concat_out  [D_MODEL];

  // Counters
  logic [$clog2(D_MODEL):0]     dim_idx;     // Output dimension being computed
  logic [$clog2(D_MODEL):0]     inner_idx;   // Inner loop (dot product accumulation)
  logic [$clog2(MAX_SEQ_LEN):0] pos_idx;
  logic [$clog2(N_HEADS):0]     head_idx;
  logic [$clog2(D_HEAD):0]      dhead_idx;   // Per-head dimension counter

  // Accumulators
  reg signed [31:0] q_acc, k_acc, v_acc, score_acc, wsum_acc, oproj_acc;

  // Softmax interface
  logic  sm_start, sm_valid;
  logic signed [MAX_SEQ_LEN-1:0][DATA_WIDTH-1:0] sm_scores_in;
  logic signed [MAX_SEQ_LEN-1:0][DATA_WIDTH-1:0] sm_probs_out;
  logic [$clog2(N_HEADS):0] sm_head_idx;

  localparam signed [15:0] SCALE_FACTOR = 16'sh0040; // 1/sqrt(16) = 0.25
  localparam signed [15:0] NEG_INF = -16'sh0800;     // -8.0

  // =========================================================================
  // Softmax Unit
  // =========================================================================
  softmax_unit #(.VEC_LEN(MAX_SEQ_LEN)) u_softmax (
    .clk(clk), .rst_n(rst_n),
    .start(sm_start), .scores(sm_scores_in),
    .probs(sm_probs_out), .valid(sm_valid)
  );

  // =========================================================================
  // BRAM Address Generation (active one cycle ahead)
  // =========================================================================
  // For QKV projection: address = inner_idx * D_MODEL + dim_idx (row-major)
  // For output projection: same pattern with concat_out
  always @(*) begin
    wqkv_rd_addr = 0;
    wqkv_rd_en   = 1'b0;
    wo_rd_addr   = 0;
    wo_rd_en     = 1'b0;

    // QKV BRAM reads: row = inner_idx, col = dim_idx
    if (state == S_PROJ_ADDR || state == S_PROJ_QKV) begin
      wqkv_rd_addr = inner_idx[$clog2(D_MODEL)-1:0] * D_MODEL + dim_idx[$clog2(D_MODEL)-1:0];
      wqkv_rd_en   = 1'b1;
    end

    // Output projection BRAM reads: row = inner_idx, col = dim_idx
    if (state == S_OPROJ_ADDR || state == S_OUTPUT_PROJ) begin
      wo_rd_addr = inner_idx[$clog2(D_MODEL)-1:0] * D_MODEL + dim_idx[$clog2(D_MODEL)-1:0];
      wo_rd_en   = 1'b1;
    end
  end

  // KV-cache address generation
  always @(*) begin
    kcache_rd_pos = 0;
    kcache_rd_dim = 0;
    vcache_rd_pos = 0;
    vcache_rd_dim = 0;

    // Score computation: read k_cache[pos_idx][head*D_HEAD + dhead_idx]
    if (state == S_SCORE_ADDR || state == S_SCORE) begin
      kcache_rd_pos = pos_idx[$clog2(MAX_SEQ_LEN)-1:0];
      kcache_rd_dim = head_idx[$clog2(N_HEADS)-1:0] * D_HEAD
                    + dhead_idx[$clog2(D_HEAD)-1:0];
    end

    // Weighted sum: read v_cache[pos_idx][head*D_HEAD + dhead_idx]
    if (state == S_WSUM_ADDR || state == S_WEIGHTED_SUM) begin
      vcache_rd_pos = pos_idx[$clog2(MAX_SEQ_LEN)-1:0];
      vcache_rd_dim = head_idx[$clog2(N_HEADS)-1:0] * D_HEAD
                    + dhead_idx[$clog2(D_HEAD)-1:0];
    end
  end

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      valid       <= 1'b0;
      cache_wr_en <= 1'b0;
      dim_idx     <= 0;
      inner_idx   <= 0;
      pos_idx     <= 0;
      head_idx    <= 0;
      dhead_idx   <= 0;
      sm_head_idx <= 0;
      sm_start    <= 1'b0;
      q_acc       <= 0;
      k_acc       <= 0;
      v_acc       <= 0;
      score_acc   <= 0;
      wsum_acc    <= 0;
      oproj_acc   <= 0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        q_vec[i]      <= 0;
        k_vec[i]      <= 0;
        v_vec[i]      <= 0;
        attn_out[i]   <= 0;
        concat_out[i] <= 0;
        k_cache_wr[i] <= 0;
        v_cache_wr[i] <= 0;
      end
      for (t = 0; t < MAX_SEQ_LEN; t = t + 1)
        sm_scores_in[t] <= 0;
    end else begin
      state       <= state_next;
      sm_start    <= 1'b0;
      cache_wr_en <= 1'b0;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            dim_idx   <= 0;
            inner_idx <= 0;
            q_acc     <= 0;
            k_acc     <= 0;
            v_acc     <= 0;
          end
        end

        // Pre-issue first BRAM address (inner_idx=0, dim_idx=0)
        S_PROJ_ADDR: begin
          inner_idx <= inner_idx + 1; // Will be 1 next cycle; data for 0 arrives
        end

        // Accumulate QKV: one MAC per cycle using BRAM data
        // Data arriving this cycle is for address issued last cycle (inner_idx-1)
        S_PROJ_QKV: begin
          q_acc <= q_acc + to_acc(x_in[inner_idx - 1]) * to_acc(wq_rd_data);
          k_acc <= k_acc + to_acc(x_in[inner_idx - 1]) * to_acc(wk_rd_data);
          v_acc <= v_acc + to_acc(x_in[inner_idx - 1]) * to_acc(wv_rd_data);

          if (inner_idx < D_MODEL)
            inner_idx <= inner_idx + 1;
        end

        // Store completed dimension, advance to next
        S_PROJ_STORE: begin
          q_vec[dim_idx] <= to_data(q_acc >>> FRAC_BITS);
          k_vec[dim_idx] <= to_data(k_acc >>> FRAC_BITS);
          v_vec[dim_idx] <= to_data(v_acc >>> FRAC_BITS);
          dim_idx   <= dim_idx + 1;
          inner_idx <= 0;
          q_acc     <= 0;
          k_acc     <= 0;
          v_acc     <= 0;
        end

        S_WRITE_CACHE: begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            k_cache_wr[i] <= k_vec[i];
            v_cache_wr[i] <= v_vec[i];
          end
          cache_wr_en <= 1'b1;
          head_idx    <= 0;
          pos_idx     <= 0;
          dhead_idx   <= 0;
          score_acc   <= 0;
        end

        // Pre-issue first k_cache read address
        S_SCORE_ADDR: begin
          dhead_idx <= dhead_idx + 1;
        end

        // Accumulate score dot product one element at a time
        // Read k_cache[pos_idx][head*D_HEAD + dhead_idx-1]
        S_SCORE: begin
          score_acc <= score_acc + to_acc(q_vec[(head_idx) * D_HEAD + (dhead_idx) - 1])
                                * to_acc(kcache_rd_data);
          if (dhead_idx < D_HEAD)
            dhead_idx <= dhead_idx + 1;
        end

        // Store completed score
        S_SCORE_STORE: begin
          head_scores[head_idx][pos_idx] <= fp_mul(to_data(score_acc >>> FRAC_BITS), SCALE_FACTOR);
          score_acc <= 0;
          dhead_idx <= 0;

          if (pos_idx < {1'b0, seq_pos}) begin
            pos_idx <= pos_idx + 1;
          end else begin
            // Next head or done
            pos_idx  <= 0;
            head_idx <= head_idx + 1;
            if (head_idx >= N_HEADS - 1)
              sm_head_idx <= 0;
          end
        end

        // Softmax states (unchanged from original)
        S_SOFTMAX_PREP: begin
          for (t = 0; t < MAX_SEQ_LEN; t = t + 1) begin
            if (t <= (seq_pos))
              sm_scores_in[t] <= head_scores[sm_head_idx][t];
            else
              sm_scores_in[t] <= NEG_INF;
          end
          sm_start <= 1'b1;
        end

        S_SOFTMAX_RUN: ;

        S_SOFTMAX_STORE: begin
          for (t = 0; t < MAX_SEQ_LEN; t = t + 1)
            head_probs[sm_head_idx][t] <= sm_probs_out[t];
          sm_head_idx <= sm_head_idx + 1;
          if (sm_head_idx >= N_HEADS - 1) begin
            head_idx  <= 0;
            pos_idx   <= 0;
            dhead_idx <= 0;
            wsum_acc  <= 0;
          end
        end

        // Pre-issue first v_cache read
        S_WSUM_ADDR: begin
          pos_idx <= pos_idx + 1;
        end

        // Accumulate weighted sum: sum_t(probs[t] * v_cache[t][head*D_HEAD+d])
        // Serialised over positions for each (head, d)
        S_WEIGHTED_SUM: begin
          wsum_acc <= wsum_acc + to_acc(head_probs[head_idx][pos_idx - 1])
                              * to_acc(vcache_rd_data);
          if (pos_idx <= {1'b0, seq_pos})
            pos_idx <= pos_idx + 1;
        end

        // Store weighted sum element
        S_WSUM_STORE: begin
          head_out[head_idx][dhead_idx] <= to_data(wsum_acc >>> FRAC_BITS);
          wsum_acc <= 0;
          pos_idx  <= 0;

          if (dhead_idx < D_HEAD - 1) begin
            dhead_idx <= dhead_idx + 1;
          end else begin
            dhead_idx <= 0;
            head_idx  <= head_idx + 1;
          end
        end

        // Concatenate heads
        S_CONCAT: begin
          for (h = 0; h < N_HEADS; h = h + 1)
            for (d = 0; d < D_HEAD; d = d + 1)
              concat_out[h * D_HEAD + d] <= head_out[h][d];
          dim_idx   <= 0;
          inner_idx <= 0;
          oproj_acc <= 0;
        end

        // Pre-issue first wo address
        S_OPROJ_ADDR: begin
          inner_idx <= inner_idx + 1;
        end

        // Accumulate output projection
        S_OUTPUT_PROJ: begin
          oproj_acc <= oproj_acc + to_acc(concat_out[inner_idx - 1]) * to_acc(wo_rd_data);
          if (inner_idx < D_MODEL)
            inner_idx <= inner_idx + 1;
        end

        // Store output dimension
        S_OPROJ_STORE: begin
          attn_out[dim_idx] <= to_data(oproj_acc >>> FRAC_BITS);
          dim_idx   <= dim_idx + 1;
          inner_idx <= 0;
          oproj_acc <= 0;
          if (dim_idx >= D_MODEL - 1)
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
      S_IDLE:          if (start) state_next = S_PROJ_ADDR;
      S_PROJ_ADDR:     state_next = S_PROJ_QKV;
      S_PROJ_QKV:      if (inner_idx >= D_MODEL)
                         state_next = S_PROJ_STORE;
      S_PROJ_STORE:    if (dim_idx >= D_MODEL - 1)
                         state_next = S_WRITE_CACHE;
                       else
                         state_next = S_PROJ_ADDR;
      S_WRITE_CACHE:   state_next = S_SCORE_ADDR;
      S_SCORE_ADDR:    state_next = S_SCORE;
      S_SCORE:         if (dhead_idx >= D_HEAD)
                         state_next = S_SCORE_STORE;
      S_SCORE_STORE:   if (head_idx >= N_HEADS - 1
                           && pos_idx >= {1'b0, seq_pos})
                         state_next = S_SOFTMAX_PREP;
                       else
                         state_next = S_SCORE_ADDR;
      S_SOFTMAX_PREP:  state_next = S_SOFTMAX_RUN;
      S_SOFTMAX_RUN:   if (sm_valid) state_next = S_SOFTMAX_STORE;
      S_SOFTMAX_STORE: if (sm_head_idx >= N_HEADS - 1)
                         state_next = S_WSUM_ADDR;
                       else
                         state_next = S_SOFTMAX_PREP;
      S_WSUM_ADDR:     state_next = S_WEIGHTED_SUM;
      S_WEIGHTED_SUM:  if (pos_idx > {1'b0, seq_pos})
                         state_next = S_WSUM_STORE;
      S_WSUM_STORE:    if (head_idx >= N_HEADS - 1
                           && dhead_idx >= D_HEAD - 1)
                         state_next = S_CONCAT;
                       else
                         state_next = S_WSUM_ADDR;
      S_CONCAT:        state_next = S_OPROJ_ADDR;
      S_OPROJ_ADDR:    state_next = S_OUTPUT_PROJ;
      S_OUTPUT_PROJ:   if (inner_idx >= D_MODEL)
                         state_next = S_OPROJ_STORE;
      S_OPROJ_STORE:   if (dim_idx >= D_MODEL - 1)
                         state_next = S_DONE;
                       else
                         state_next = S_OPROJ_ADDR;
      S_DONE:          state_next = S_IDLE;
      default:         state_next = S_IDLE;
    endcase
  end

endmodule
