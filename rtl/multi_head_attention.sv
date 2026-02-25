// =============================================================================
// multi_head_attention.sv - Multi-Head Scaled Dot-Product Attention
// =============================================================================
// Implements the core attention mechanism of the transformer decoder:
//
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// For decoder (causal) inference, this operates on a single query token
// against the full KV-cache. The module:
//   1. Projects input x to Q, K, V via linear layers (via systolic array)
//   2. Splits into N_HEADS parallel heads
//   3. Computes scaled dot-product attention per head
//   4. Concatenates heads and projects output
//
// KV-Cache: Stores K and V vectors for all previous positions to enable
// autoregressive generation without recomputation.
// =============================================================================

module multi_head_attention
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  // Input token embedding
  input  data_t  x_in [D_MODEL],

  // Weight matrices (stored externally, streamed in)
  input  data_t  wq [D_MODEL][D_MODEL],  // Query projection
  input  data_t  wk [D_MODEL][D_MODEL],  // Key projection
  input  data_t  wv [D_MODEL][D_MODEL],  // Value projection
  input  data_t  wo [D_MODEL][D_MODEL],  // Output projection

  // KV Cache interface
  input  seq_idx_t              seq_pos,        // Current position in sequence
  output data_t                 k_cache_wr [D_MODEL],
  output data_t                 v_cache_wr [D_MODEL],
  output logic                  cache_wr_en,
  input  data_t                 k_cache [MAX_SEQ_LEN][D_MODEL],
  input  data_t                 v_cache [MAX_SEQ_LEN][D_MODEL],

  // Output
  output data_t  attn_out [D_MODEL],
  output logic   valid
);

  // =========================================================================
  // State Machine
  // =========================================================================
  typedef enum logic [3:0] {
    S_IDLE,
    S_PROJ_QKV,
    S_WRITE_CACHE,
    S_SCORE,
    S_SOFTMAX_WAIT,
    S_WEIGHTED_SUM,
    S_OUTPUT_PROJ,
    S_DONE
  } state_t;

  state_t state, state_next;

  // Internal signals
  data_t q_vec [D_MODEL];
  data_t k_vec [D_MODEL];
  data_t v_vec [D_MODEL];

  // Per-head attention scores and outputs
  data_t head_scores [N_HEADS][MAX_SEQ_LEN];
  data_t head_probs  [N_HEADS][MAX_SEQ_LEN];
  data_t head_out    [N_HEADS][D_HEAD];
  data_t concat_out  [D_MODEL];

  // Counters
  logic [$clog2(D_MODEL):0]     dim_idx;
  logic [$clog2(MAX_SEQ_LEN):0] pos_idx;
  logic [$clog2(N_HEADS):0]     head_idx;

  // Softmax control
  logic softmax_start;
  logic softmax_done;

  // Scale factor: 1/sqrt(D_HEAD) in Q8.8 ≈ 1/4 = 0x0040
  localparam data_t SCALE_FACTOR = 16'sh0040;

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      valid        <= 1'b0;
      cache_wr_en  <= 1'b0;
      dim_idx      <= '0;
      pos_idx      <= '0;
      head_idx     <= '0;
      softmax_start<= 1'b0;
      for (int i = 0; i < D_MODEL; i++) begin
        q_vec[i]      <= '0;
        k_vec[i]      <= '0;
        v_vec[i]      <= '0;
        attn_out[i]   <= '0;
        concat_out[i] <= '0;
        k_cache_wr[i] <= '0;
        v_cache_wr[i] <= '0;
      end
    end else begin
      state <= state_next;
      softmax_start <= 1'b0;
      cache_wr_en   <= 1'b0;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            dim_idx <= '0;
          end
        end

        // Linear projection: Q = x * Wq, K = x * Wk, V = x * Wv
        S_PROJ_QKV: begin
          if (dim_idx < D_MODEL[$clog2(D_MODEL):0]) begin
            acc_t q_acc, k_acc, v_acc;
            q_acc = '0; k_acc = '0; v_acc = '0;
            for (int j = 0; j < D_MODEL; j++) begin
              q_acc = q_acc + acc_t'(x_in[j]) * acc_t'(wq[j][dim_idx]);
              k_acc = k_acc + acc_t'(x_in[j]) * acc_t'(wk[j][dim_idx]);
              v_acc = v_acc + acc_t'(x_in[j]) * acc_t'(wv[j][dim_idx]);
            end
            q_vec[dim_idx] <= data_t'(q_acc >>> FRAC_BITS);
            k_vec[dim_idx] <= data_t'(k_acc >>> FRAC_BITS);
            v_vec[dim_idx] <= data_t'(v_acc >>> FRAC_BITS);
            dim_idx <= dim_idx + 1;
          end
        end

        S_WRITE_CACHE: begin
          for (int i = 0; i < D_MODEL; i++) begin
            k_cache_wr[i] <= k_vec[i];
            v_cache_wr[i] <= v_vec[i];
          end
          cache_wr_en <= 1'b1;
          head_idx    <= '0;
          pos_idx     <= '0;
        end

        // Compute attention scores: score[h][t] = Q_h . K_h[t] / sqrt(d_k)
        S_SCORE: begin
          if (head_idx < N_HEADS[$clog2(N_HEADS):0]) begin
            if (pos_idx <= {1'b0, seq_pos}) begin
              acc_t dot = '0;
              for (int d = 0; d < D_HEAD; d++) begin
                int qi = int'(head_idx) * D_HEAD + d;
                int ki = int'(head_idx) * D_HEAD + d;
                dot = dot + acc_t'(q_vec[qi]) * acc_t'(k_cache[pos_idx][ki]);
              end
              head_scores[head_idx][pos_idx] <= fp_mul(data_t'(dot >>> FRAC_BITS), SCALE_FACTOR);
              pos_idx <= pos_idx + 1;
            end else begin
              // Causal mask: set future positions to large negative
              head_idx <= head_idx + 1;
              pos_idx  <= '0;
            end
          end else begin
            softmax_start <= 1'b1;
          end
        end

        S_SOFTMAX_WAIT: begin
          // Wait for softmax completion (simplified: direct copy for now)
          // In full implementation, instantiate softmax_unit per head
          for (int h = 0; h < N_HEADS; h++) begin
            for (int t = 0; t < MAX_SEQ_LEN; t++) begin
              if (t <= int'(seq_pos))
                head_probs[h][t] <= head_scores[h][t]; // Simplified
              else
                head_probs[h][t] <= '0;
            end
          end
          head_idx <= '0;
          dim_idx  <= '0;
        end

        // Weighted sum: out_h = sum_t(probs[t] * V_h[t])
        S_WEIGHTED_SUM: begin
          if (head_idx < N_HEADS[$clog2(N_HEADS):0]) begin
            for (int d = 0; d < D_HEAD; d++) begin
              acc_t ws = '0;
              for (int t = 0; t <= int'(seq_pos); t++) begin
                int vi = int'(head_idx) * D_HEAD + d;
                ws = ws + acc_t'(head_probs[head_idx][t]) * acc_t'(v_cache[t][vi]);
              end
              head_out[head_idx][d] <= data_t'(ws >>> FRAC_BITS);
            end
            head_idx <= head_idx + 1;
          end else begin
            // Concatenate heads
            for (int h = 0; h < N_HEADS; h++)
              for (int d = 0; d < D_HEAD; d++)
                concat_out[h * D_HEAD + d] <= head_out[h][d];
            dim_idx <= '0;
          end
        end

        // Output projection: attn_out = concat * Wo
        S_OUTPUT_PROJ: begin
          if (dim_idx < D_MODEL[$clog2(D_MODEL):0]) begin
            acc_t o_acc = '0;
            for (int j = 0; j < D_MODEL; j++)
              o_acc = o_acc + acc_t'(concat_out[j]) * acc_t'(wo[j][dim_idx]);
            attn_out[dim_idx] <= data_t'(o_acc >>> FRAC_BITS);
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
  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:          if (start) state_next = S_PROJ_QKV;
      S_PROJ_QKV:      if (dim_idx >= D_MODEL[$clog2(D_MODEL):0]) state_next = S_WRITE_CACHE;
      S_WRITE_CACHE:   state_next = S_SCORE;
      S_SCORE:         if (head_idx >= N_HEADS[$clog2(N_HEADS):0]) state_next = S_SOFTMAX_WAIT;
      S_SOFTMAX_WAIT:  state_next = S_WEIGHTED_SUM;
      S_WEIGHTED_SUM:  if (head_idx >= N_HEADS[$clog2(N_HEADS):0]) state_next = S_OUTPUT_PROJ;
      S_OUTPUT_PROJ:   if (dim_idx >= D_MODEL[$clog2(D_MODEL):0]) state_next = S_DONE;
      S_DONE:          state_next = S_IDLE;
      default:         state_next = S_IDLE;
    endcase
  end

endmodule
