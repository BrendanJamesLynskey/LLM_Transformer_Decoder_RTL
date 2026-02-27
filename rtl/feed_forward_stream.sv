// =============================================================================
// feed_forward_stream.sv - Streaming Feed-Forward Network
// =============================================================================
// Streaming variant of feed_forward that reads weights from BRAM via
// address/data interfaces instead of combinational array ports.
//
// FFN(x) = ReLU(x * W1 + b1) * W2 + b2
//
// Weight access pattern:
//   Linear1: for each hidden dim h, read w1[j][h] for j=0..D_MODEL-1
//     BRAM address = j * D_FF + h (row-major)
//   Linear2: for each output dim d, read w2[j][d] for j=0..D_FF-1
//     BRAM address = j * D_MODEL + d (row-major)
//   Biases: b1[h] read once per hidden dim, b2[d] once per output dim
//
// Dot products are fully serialised: one MAC per cycle.
// =============================================================================

module feed_forward_stream
  import transformer_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   start,

  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] x_in,

  // W1 BRAM read interface (D_MODEL × D_FF)
  output logic [$clog2(D_MODEL*D_FF)-1:0] w1_rd_addr,
  input  data_t  w1_rd_data,
  output logic   w1_rd_en,

  // b1 BRAM read interface (D_FF)
  output logic [$clog2(D_FF)-1:0] b1_rd_addr,
  input  data_t  b1_rd_data,
  output logic   b1_rd_en,

  // W2 BRAM read interface (D_FF × D_MODEL)
  output logic [$clog2(D_FF*D_MODEL)-1:0] w2_rd_addr,
  input  data_t  w2_rd_data,
  output logic   w2_rd_en,

  // b2 read: small enough to keep as packed port (D_MODEL elements)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2,

  output logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] y_out,
  output logic   valid
);

  typedef enum logic [3:0] {
    S_IDLE,
    S_L1_B1_ADDR,    // Pre-issue b1 read
    S_L1_ADDR,       // Pre-issue first w1 BRAM address
    S_LINEAR1,       // Accumulate dot product
    S_L1_STORE,      // Store hidden[ff_idx] = acc + b1, advance
    S_RELU,
    S_L2_ADDR,       // Pre-issue first w2 BRAM address
    S_LINEAR2,       // Accumulate dot product
    S_L2_STORE,      // Store y_out[out_idx] = acc + b2, advance
    S_DONE
  } state_t;

  state_t state, state_next;

  data_t hidden [D_FF];
  logic [$clog2(D_FF):0]    ff_idx;
  logic [$clog2(D_MODEL):0] out_idx;
  logic [$clog2(D_FF):0]    inner_idx;  // Inner loop counter for dot product
  acc_t  acc;
  data_t b1_latched;  // Latched b1 value (read one cycle ahead)

  // =========================================================================
  // BRAM Address Generation
  // =========================================================================
  always_comb begin
    w1_rd_addr = '0;
    w1_rd_en   = 1'b0;
    b1_rd_addr = '0;
    b1_rd_en   = 1'b0;
    w2_rd_addr = '0;
    w2_rd_en   = 1'b0;

    // b1 pre-read (one cycle before LINEAR1 starts)
    if (state == S_L1_B1_ADDR) begin
      b1_rd_addr = ff_idx[$clog2(D_FF)-1:0];
      b1_rd_en   = 1'b1;
    end

    // W1 reads: row = inner_idx, col = ff_idx
    if (state == S_L1_ADDR || state == S_LINEAR1) begin
      w1_rd_addr = inner_idx[$clog2(D_MODEL)-1:0] * D_FF[8:0] + ff_idx[$clog2(D_FF)-1:0];
      w1_rd_en   = 1'b1;
    end

    // W2 reads: row = inner_idx, col = out_idx
    if (state == S_L2_ADDR || state == S_LINEAR2) begin
      w2_rd_addr = inner_idx[$clog2(D_FF)-1:0] * D_MODEL[6:0] + out_idx[$clog2(D_MODEL)-1:0];
      w2_rd_en   = 1'b1;
    end
  end

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      valid     <= 1'b0;
      ff_idx    <= '0;
      out_idx   <= '0;
      inner_idx <= '0;
      acc       <= '0;
      b1_latched <= '0;
      for (int i = 0; i < D_FF; i++)
        hidden[i] <= '0;
      for (int i = 0; i < D_MODEL; i++)
        y_out[i] <= '0;
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            ff_idx    <= '0;
            out_idx   <= '0;
            inner_idx <= '0;
            acc       <= '0;
          end
        end

        // Pre-read b1[ff_idx]
        S_L1_B1_ADDR: ;

        // Pre-issue first w1 address, latch b1
        S_L1_ADDR: begin
          b1_latched <= b1_rd_data;
          inner_idx  <= inner_idx + 1;
        end

        // Accumulate: acc += x_in[inner_idx-1] * w1_rd_data
        S_LINEAR1: begin
          acc <= acc + acc_t'(x_in[inner_idx - 1]) * acc_t'(w1_rd_data);
          if (inner_idx < D_MODEL[$clog2(D_MODEL):0])
            inner_idx <= inner_idx + 1;
        end

        // Store hidden[ff_idx] = truncate(acc) + b1
        S_L1_STORE: begin
          hidden[ff_idx] <= data_t'(acc >>> FRAC_BITS) + b1_latched;
          ff_idx    <= ff_idx + 1;
          inner_idx <= '0;
          acc       <= '0;
        end

        // ReLU
        S_RELU: begin
          for (int i = 0; i < D_FF; i++)
            if (hidden[i][DATA_WIDTH-1])
              hidden[i] <= '0;
          out_idx   <= '0;
          inner_idx <= '0;
          acc       <= '0;
        end

        // Pre-issue first w2 address
        S_L2_ADDR: begin
          inner_idx <= inner_idx + 1;
        end

        // Accumulate: acc += hidden[inner_idx-1] * w2_rd_data
        S_LINEAR2: begin
          acc <= acc + acc_t'(hidden[inner_idx - 1]) * acc_t'(w2_rd_data);
          if (inner_idx < D_FF[$clog2(D_FF):0])
            inner_idx <= inner_idx + 1;
        end

        // Store y_out[out_idx] = truncate(acc) + b2
        S_L2_STORE: begin
          y_out[out_idx] <= data_t'(acc >>> FRAC_BITS) + b2[out_idx];
          out_idx   <= out_idx + 1;
          inner_idx <= '0;
          acc       <= '0;
          if (out_idx >= D_MODEL[$clog2(D_MODEL):0] - 1)
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
  always_comb begin
    state_next = state;
    case (state)
      S_IDLE:      if (start) state_next = S_L1_B1_ADDR;
      S_L1_B1_ADDR: state_next = S_L1_ADDR;
      S_L1_ADDR:   state_next = S_LINEAR1;
      S_LINEAR1:   if (inner_idx >= D_MODEL[$clog2(D_MODEL):0])
                     state_next = S_L1_STORE;
      S_L1_STORE:  if (ff_idx >= D_FF[$clog2(D_FF):0] - 1)
                     state_next = S_RELU;
                   else
                     state_next = S_L1_B1_ADDR;
      S_RELU:      state_next = S_L2_ADDR;
      S_L2_ADDR:   state_next = S_LINEAR2;
      S_LINEAR2:   if (inner_idx >= D_FF[$clog2(D_FF):0])
                     state_next = S_L2_STORE;
      S_L2_STORE:  if (out_idx >= D_MODEL[$clog2(D_MODEL):0] - 1)
                     state_next = S_DONE;
                   else
                     state_next = S_L2_ADDR;
      S_DONE:      state_next = S_IDLE;
      default:     state_next = S_IDLE;
    endcase
  end

endmodule
