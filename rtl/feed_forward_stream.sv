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
  output reg [$clog2(D_MODEL*D_FF)-1:0] w1_rd_addr,
  input  signed [15:0]  w1_rd_data,
  output reg   w1_rd_en,

  // b1 BRAM read interface (D_FF)
  output reg [$clog2(D_FF)-1:0] b1_rd_addr,
  input  signed [15:0]  b1_rd_data,
  output reg   b1_rd_en,

  // W2 BRAM read interface (D_FF × D_MODEL)
  output reg [$clog2(D_FF*D_MODEL)-1:0] w2_rd_addr,
  input  signed [15:0]  w2_rd_data,
  output reg   w2_rd_en,

  // b2 read: small enough to keep as packed port (D_MODEL elements)
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] b2,

  output reg signed [D_MODEL-1:0][DATA_WIDTH-1:0] y_out,
  output reg   valid
);

  integer i;
  localparam [3:0] S_IDLE = 0;
  localparam [3:0] S_L1_B1_ADDR = 1;
  localparam [3:0] S_L1_ADDR = 2;
  localparam [3:0] S_LINEAR1 = 3;
  localparam [3:0] S_L1_STORE = 4;
  localparam [3:0] S_RELU = 5;
  localparam [3:0] S_L2_ADDR = 6;
  localparam [3:0] S_LINEAR2 = 7;
  localparam [3:0] S_L2_STORE = 8;
  localparam [3:0] S_DONE = 9;

  reg [3:0] state, state_next;

  reg signed [15:0] hidden [D_FF];
  logic [$clog2(D_FF):0]    ff_idx;
  logic [$clog2(D_MODEL):0] out_idx;
  logic [$clog2(D_FF):0]    inner_idx;  // Inner loop counter for dot product
  reg signed [31:0]  acc;
  reg signed [15:0] b1_latched;  // Latched b1 value (read one cycle ahead)

  // =========================================================================
  // BRAM Address Generation
  // =========================================================================
  always @(*) begin
    w1_rd_addr = 0;
    w1_rd_en   = 1'b0;
    b1_rd_addr = 0;
    b1_rd_en   = 1'b0;
    w2_rd_addr = 0;
    w2_rd_en   = 1'b0;

    // b1 pre-read (one cycle before LINEAR1 starts)
    if (state == S_L1_B1_ADDR) begin
      b1_rd_addr = ff_idx[$clog2(D_FF)-1:0];
      b1_rd_en   = 1'b1;
    end

    // W1 reads: row = inner_idx, col = ff_idx
    if (state == S_L1_ADDR || state == S_LINEAR1) begin
      w1_rd_addr = inner_idx[$clog2(D_MODEL)-1:0] * D_FF + ff_idx[$clog2(D_FF)-1:0];
      w1_rd_en   = 1'b1;
    end

    // W2 reads: row = inner_idx, col = out_idx
    if (state == S_L2_ADDR || state == S_LINEAR2) begin
      w2_rd_addr = inner_idx[$clog2(D_FF)-1:0] * D_MODEL + out_idx[$clog2(D_MODEL)-1:0];
      w2_rd_en   = 1'b1;
    end
  end

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      valid     <= 1'b0;
      ff_idx    <= 0;
      out_idx   <= 0;
      inner_idx <= 0;
      acc       <= 0;
      b1_latched <= 0;
      for (i = 0; i < D_FF; i = i + 1)
        hidden[i] <= 0;
      for (i = 0; i < D_MODEL; i = i + 1)
        y_out[i] <= 0;
    end else begin
      state <= state_next;

      case (state)
        S_IDLE: begin
          valid <= 1'b0;
          if (start) begin
            ff_idx    <= 0;
            out_idx   <= 0;
            inner_idx <= 0;
            acc       <= 0;
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
          acc <= acc + to_acc(x_in[inner_idx - 1]) * to_acc(w1_rd_data);
          if (inner_idx < D_MODEL)
            inner_idx <= inner_idx + 1;
        end

        // Store hidden[ff_idx] = truncate(acc) + b1
        S_L1_STORE: begin
          hidden[ff_idx] <= to_data(acc >>> FRAC_BITS) + b1_latched;
          ff_idx    <= ff_idx + 1;
          inner_idx <= 0;
          acc       <= 0;
        end

        // ReLU
        S_RELU: begin
          for (i = 0; i < D_FF; i = i + 1)
            if (hidden[i][DATA_WIDTH-1])
              hidden[i] <= 0;
          out_idx   <= 0;
          inner_idx <= 0;
          acc       <= 0;
        end

        // Pre-issue first w2 address
        S_L2_ADDR: begin
          inner_idx <= inner_idx + 1;
        end

        // Accumulate: acc += hidden[inner_idx-1] * w2_rd_data
        S_LINEAR2: begin
          acc <= acc + to_acc(hidden[inner_idx - 1]) * to_acc(w2_rd_data);
          if (inner_idx < D_FF)
            inner_idx <= inner_idx + 1;
        end

        // Store y_out[out_idx] = truncate(acc) + b2
        S_L2_STORE: begin
          y_out[out_idx] <= to_data(acc >>> FRAC_BITS) + b2[out_idx];
          out_idx   <= out_idx + 1;
          inner_idx <= 0;
          acc       <= 0;
          if (out_idx >= D_MODEL - 1)
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
      S_IDLE:      if (start) state_next = S_L1_B1_ADDR;
      S_L1_B1_ADDR: state_next = S_L1_ADDR;
      S_L1_ADDR:   state_next = S_LINEAR1;
      S_LINEAR1:   if (inner_idx >= D_MODEL)
                     state_next = S_L1_STORE;
      S_L1_STORE:  if (ff_idx >= D_FF - 1)
                     state_next = S_RELU;
                   else
                     state_next = S_L1_B1_ADDR;
      S_RELU:      state_next = S_L2_ADDR;
      S_L2_ADDR:   state_next = S_LINEAR2;
      S_LINEAR2:   if (inner_idx >= D_FF)
                     state_next = S_L2_STORE;
      S_L2_STORE:  if (out_idx >= D_MODEL - 1)
                     state_next = S_DONE;
                   else
                     state_next = S_L2_ADDR;
      S_DONE:      state_next = S_IDLE;
      default:     state_next = S_IDLE;
    endcase
  end

endmodule
