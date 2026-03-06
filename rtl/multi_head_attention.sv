// =============================================================================
// multi_head_attention.sv - Multi-Head Scaled Dot-Product Attention
// =============================================================================
// Implements the core attention mechanism of the transformer decoder:
//
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// For decoder (causal) inference, this operates on a single query token
// against the full KV-cache. The module:
//   1. Projects input x to Q, K, V via linear layers
//   2. Splits into N_HEADS parallel heads
//   3. Computes scaled dot-product attention per head
//   4. Applies softmax via an instantiated softmax_unit (time-multiplexed)
//   5. Concatenates heads and projects output
//
// KV-Cache: Stores K and V vectors for all previous positions to enable
// autoregressive generation without recomputation.
//
// Softmax Integration: A single softmax_unit (VEC_LEN = MAX_SEQ_LEN) is
// time-multiplexed across heads. Positions beyond seq_pos are padded with
// a large negative value (-128.0) so softmax drives them to near-zero
// probability, implementing the causal mask.
// =============================================================================

module multi_head_attention
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  // Input token embedding (packed 1D)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] x_in,

  // Weight matrices (unpacked — set before sim, no runtime propagation needed)
  input  signed [15:0]  wq [D_MODEL][D_MODEL],  // Query projection
  input  signed [15:0]  wk [D_MODEL][D_MODEL],  // Key projection
  input  signed [15:0]  wv [D_MODEL][D_MODEL],  // Value projection
  input  signed [15:0]  wo [D_MODEL][D_MODEL],  // Output projection

  // KV Cache interface
  input  [6:0]              seq_pos,
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] k_cache_wr,
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] v_cache_wr,
  output reg                  cache_wr_en,
  input  signed [15:0]                 k_cache [MAX_SEQ_LEN][D_MODEL],
  input  signed [15:0]                 v_cache [MAX_SEQ_LEN][D_MODEL],

  // Output (packed 1D)
  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] attn_out,
  output reg   valid
);

  integer d, h, i, j, t;
  reg signed [31:0] q_acc;
  reg signed [31:0] k_acc;
  reg signed [31:0] v_acc;
  reg signed [31:0] dot;
  reg signed [31:0] ws;
  reg signed [31:0] o_acc;
  integer qi;
  integer ki;
  integer vi;
  // =========================================================================
  // State Machine
  // =========================================================================
  localparam [3:0] S_IDLE = 0;
  localparam [3:0] S_PROJ_QKV = 1;
  localparam [3:0] S_WRITE_CACHE = 2;
  localparam [3:0] S_SCORE = 3;
  localparam [3:0] S_SOFTMAX_PREP = 4;
  localparam [3:0] S_SOFTMAX_RUN = 5;
  localparam [3:0] S_SOFTMAX_STORE = 6;
  localparam [3:0] S_WEIGHTED_SUM = 7;
  localparam [3:0] S_OUTPUT_PROJ = 8;
  localparam [3:0] S_DONE = 9;

  reg [3:0] state, state_next;

  // Internal signals
  reg signed [15:0] q_vec [D_MODEL];
  reg signed [15:0] k_vec [D_MODEL];
  reg signed [15:0] v_vec [D_MODEL];

  // Per-head attention scores and outputs
  reg signed [15:0] head_scores [N_HEADS][MAX_SEQ_LEN];
  reg signed [15:0] head_probs  [N_HEADS][MAX_SEQ_LEN];
  reg signed [15:0] head_out    [N_HEADS][D_HEAD];
  reg signed [15:0] concat_out  [D_MODEL];

  // Counters
  logic [$clog2(D_MODEL):0]     dim_idx;
  logic [$clog2(MAX_SEQ_LEN):0] pos_idx;
  logic [$clog2(N_HEADS):0]     head_idx;

  // Softmax unit interface
  logic  sm_start;
  logic  sm_valid;
  logic signed [MAX_SEQ_LEN-1:0][DATA_WIDTH-1:0] sm_scores_in;  // Padded input to softmax
  logic signed [MAX_SEQ_LEN-1:0][DATA_WIDTH-1:0] sm_probs_out;  // Output from softmax

  // Softmax head counter (which head is being processed)
  logic [$clog2(N_HEADS):0] sm_head_idx;

  // Scale factor: 1/sqrt(D_HEAD) in Q8.8 = 1/4 = 0x0040
  localparam signed [15:0] SCALE_FACTOR = 16'sh0040;

  // Large negative for causal mask padding: -8.0 in Q8.8
  // Chosen to be within the PWL exp range and avoid overflow when
  // subtracted from positive scores in the softmax's max-subtract step.
  localparam signed [15:0] NEG_INF = -16'sh0800;

  // =========================================================================
  // Softmax Unit Instantiation (VEC_LEN = MAX_SEQ_LEN)
  // =========================================================================
  softmax_unit #(
    .VEC_LEN(MAX_SEQ_LEN)
  ) u_softmax (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (sm_start),
    .scores (sm_scores_in),
    .probs  (sm_probs_out),
    .valid  (sm_valid)
  );

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      valid        <= 1'b0;
      cache_wr_en  <= 1'b0;
      dim_idx      <= 0;
      pos_idx      <= 0;
      head_idx     <= 0;
      sm_head_idx  <= 0;
      sm_start     <= 1'b0;
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
      state <= state_next;
      sm_start    <= 1'b0;
      cache_wr_en <= 1'b0;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            dim_idx <= 0;
          end
        end

        // Linear projection: Q = x * Wq, K = x * Wk, V = x * Wv
        S_PROJ_QKV: begin
          if (dim_idx < D_MODEL) begin

            q_acc = 0; k_acc = 0; v_acc = 0;
            for (j = 0; j < D_MODEL; j = j + 1) begin
              q_acc = q_acc + to_acc(x_in[j]) * to_acc(wq[j][dim_idx]);
              k_acc = k_acc + to_acc(x_in[j]) * to_acc(wk[j][dim_idx]);
              v_acc = v_acc + to_acc(x_in[j]) * to_acc(wv[j][dim_idx]);
            end
            q_vec[dim_idx] <= to_data(q_acc >>> FRAC_BITS);
            k_vec[dim_idx] <= to_data(k_acc >>> FRAC_BITS);
            v_vec[dim_idx] <= to_data(v_acc >>> FRAC_BITS);
            dim_idx <= dim_idx + 1;
          end
        end

        S_WRITE_CACHE: begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            k_cache_wr[i] <= k_vec[i];
            v_cache_wr[i] <= v_vec[i];
          end
          cache_wr_en <= 1'b1;
          head_idx    <= 0;
          pos_idx     <= 0;
        end

        // Compute attention scores: score[h][t] = Q_h . K_h[t] / sqrt(d_k)
        S_SCORE: begin
          if (head_idx < N_HEADS) begin
            if (pos_idx <= {1'b0, seq_pos}) begin
              dot = 0;
              for (d = 0; d < D_HEAD; d = d + 1) begin
                qi = (head_idx) * D_HEAD + d;
                ki = (head_idx) * D_HEAD + d;
                dot = dot + to_acc(q_vec[qi]) * to_acc(k_cache[pos_idx][ki]);
              end
              head_scores[head_idx][pos_idx] <= fp_mul(to_data(dot >>> FRAC_BITS), SCALE_FACTOR);
              pos_idx <= pos_idx + 1;
            end else begin
              // Advance to next head
              head_idx <= head_idx + 1;
              pos_idx  <= 0;
            end
          end else begin
            // All scores computed, begin softmax processing
            sm_head_idx <= 0;
          end
        end

        // Prepare the softmax input vector for the current head
        // Valid positions get actual scores; future positions get NEG_INF
        S_SOFTMAX_PREP: begin
          for (t = 0; t < MAX_SEQ_LEN; t = t + 1) begin
            if (t <= (seq_pos))
              sm_scores_in[t] <= head_scores[sm_head_idx][t];
            else
              sm_scores_in[t] <= NEG_INF;
          end
          sm_start <= 1'b1;
        end

        // Wait for softmax_unit to complete
        S_SOFTMAX_RUN: begin
          // sm_start was pulsed in PREP; now just wait for sm_valid
        end

        // Store softmax results for this head, advance to next
        S_SOFTMAX_STORE: begin
          for (t = 0; t < MAX_SEQ_LEN; t = t + 1)
            head_probs[sm_head_idx][t] <= sm_probs_out[t];
          sm_head_idx <= sm_head_idx + 1;
          // Reset head_idx for weighted sum when this is the last head
          if (sm_head_idx >= (N_HEADS - 1))
            head_idx <= 0;
        end

        // Weighted sum: out_h = sum_t(probs[t] * V_h[t])
        S_WEIGHTED_SUM: begin
          if (head_idx < N_HEADS) begin
            for (d = 0; d < D_HEAD; d = d + 1) begin
              ws = 0;
              for (t = 0; t <= (seq_pos); t = t + 1) begin
                vi = (head_idx) * D_HEAD + d;
                ws = ws + to_acc(head_probs[head_idx][t]) * to_acc(v_cache[t][vi]);
              end
              head_out[head_idx][d] <= to_data(ws >>> FRAC_BITS);
            end
            head_idx <= head_idx + 1;
          end else begin
            // Concatenate heads
            for (h = 0; h < N_HEADS; h = h + 1)
              for (d = 0; d < D_HEAD; d = d + 1)
                concat_out[h * D_HEAD + d] <= head_out[h][d];
            dim_idx <= 0;
          end
        end

        // Output projection: attn_out = concat * Wo
        S_OUTPUT_PROJ: begin
          if (dim_idx < D_MODEL) begin
            o_acc = 0;
            for (j = 0; j < D_MODEL; j = j + 1)
              o_acc = o_acc + to_acc(concat_out[j]) * to_acc(wo[j][dim_idx]);
            attn_out[dim_idx] <= to_data(o_acc >>> FRAC_BITS);
            dim_idx <= dim_idx + 1;
          end else begin
            valid <= 1'b1;
          end
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
      S_IDLE:           if (start) state_next = S_PROJ_QKV;
      S_PROJ_QKV:       if (dim_idx >= D_MODEL) state_next = S_WRITE_CACHE;
      S_WRITE_CACHE:    state_next = S_SCORE;
      S_SCORE:          if (head_idx >= N_HEADS) state_next = S_SOFTMAX_PREP;
      S_SOFTMAX_PREP:   state_next = S_SOFTMAX_RUN;
      S_SOFTMAX_RUN:    if (sm_valid) state_next = S_SOFTMAX_STORE;
      S_SOFTMAX_STORE:  if (sm_head_idx >= (N_HEADS - 1))
                          state_next = S_WEIGHTED_SUM;
                        else
                          state_next = S_SOFTMAX_PREP;
      S_WEIGHTED_SUM:   if (head_idx >= N_HEADS) state_next = S_OUTPUT_PROJ;
      S_OUTPUT_PROJ:    if (dim_idx >= D_MODEL) state_next = S_DONE;
      S_DONE:           state_next = S_IDLE;
      default:          state_next = S_IDLE;
    endcase
  end

endmodule
