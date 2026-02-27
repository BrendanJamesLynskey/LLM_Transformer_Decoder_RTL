// =============================================================================
// kv_cache_bram.sv - KV-Cache Backed by Dual-Port BRAM
// =============================================================================
// Stores one cache matrix (K or V) of dimension MAX_SEQ_LEN × D_MODEL
// in a dual-port BRAM. Replaces the combinational array ports in the
// original design.
//
// Port A (write): Writes one D_MODEL-wide vector at a given sequence
//   position. The write is performed element-by-element over D_MODEL
//   cycles: assert wr_start, and the module auto-increments through
//   all D_MODEL dimensions, reading from wr_vec[dim].
//
// Port B (read): Random-access read of individual elements at
//   (position, dimension). Single-cycle address, data available next
//   cycle. Used by the attention scoring loop to read one element
//   per cycle.
//
// Initialisation: pass INIT_FILE to pre-load from hex (e.g. for test
//   vectors). Default is zero-initialised.
//
// Address mapping: mem[position * D_MODEL + dim] = cache[position][dim]
// =============================================================================

module kv_cache_bram
  import transformer_pkg::*;
#(
  parameter INIT_FILE = ""
)(
  input  logic                              clk,
  input  logic                              rst_n,

  // --- Write interface (port A): write a full vector at seq_pos ---
  input  logic                              wr_start,     // Pulse to begin write
  input  seq_idx_t                          wr_seq_pos,   // Which position to write
  input  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] wr_vec, // Vector to write (packed)

  output logic                              wr_done,      // Pulses when write complete

  // --- Element read interface (port B) ---
  input  seq_idx_t                          rd_pos,       // Sequence position to read
  input  logic [$clog2(D_MODEL)-1:0]        rd_dim,       // Dimension to read
  output data_t                             rd_data       // Data (1-cycle latency)
);

  localparam int DEPTH  = MAX_SEQ_LEN * D_MODEL;
  localparam int ADDR_W = $clog2(DEPTH);

  // Dual-port BRAM signals
  logic                  a_we;
  logic [ADDR_W-1:0]     a_addr;
  logic [DATA_WIDTH-1:0] a_wdata;
  logic [DATA_WIDTH-1:0] a_rdata; // unused for write port
  logic                  b_we;
  logic [ADDR_W-1:0]     b_addr;
  logic [DATA_WIDTH-1:0] b_wdata;
  logic [DATA_WIDTH-1:0] b_rdata;

  bram_dp #(
    .DATA_WIDTH (DATA_WIDTH),
    .DEPTH      (DEPTH),
    .INIT_FILE  (INIT_FILE)
  ) u_bram (
    .clk     (clk),
    .a_we    (a_we),
    .a_addr  (a_addr),
    .a_wdata (a_wdata),
    .a_rdata (a_rdata),
    .b_we    (b_we),
    .b_addr  (b_addr),
    .b_wdata (b_wdata),
    .b_rdata (b_rdata)
  );

  // =========================================================================
  // Write FSM — writes D_MODEL elements sequentially via port A
  // =========================================================================
  typedef enum logic [1:0] {
    WR_IDLE,
    WR_ACTIVE,
    WR_DONE
  } wr_state_t;

  wr_state_t wr_state, wr_state_next;
  logic [$clog2(D_MODEL):0] wr_dim_idx;
  seq_idx_t wr_pos_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_state   <= WR_IDLE;
      wr_dim_idx <= '0;
      wr_pos_reg <= '0;
      wr_done    <= 1'b0;
    end else begin
      wr_state <= wr_state_next;
      wr_done  <= 1'b0;

      case (wr_state)
        WR_IDLE: begin
          if (wr_start) begin
            wr_pos_reg <= wr_seq_pos;
            wr_dim_idx <= '0;
          end
        end

        WR_ACTIVE: begin
          if (wr_dim_idx < D_MODEL[$clog2(D_MODEL):0]) begin
            wr_dim_idx <= wr_dim_idx + 1;
          end else begin
            wr_done <= 1'b1;
          end
        end

        WR_DONE: ;
      endcase
    end
  end

  always_comb begin
    wr_state_next = wr_state;
    case (wr_state)
      WR_IDLE:   if (wr_start) wr_state_next = WR_ACTIVE;
      WR_ACTIVE: if (wr_dim_idx >= D_MODEL[$clog2(D_MODEL):0]) wr_state_next = WR_DONE;
      WR_DONE:   wr_state_next = WR_IDLE;
      default:   wr_state_next = WR_IDLE;
    endcase
  end

  // Port A: write address generation
  always_comb begin
    if (wr_state == WR_ACTIVE && wr_dim_idx < D_MODEL[$clog2(D_MODEL):0]) begin
      a_we    = 1'b1;
      a_addr  = ADDR_W'({1'b0, wr_pos_reg} * D_MODEL + wr_dim_idx);
      a_wdata = wr_vec[wr_dim_idx];
    end else begin
      a_we    = 1'b0;
      a_addr  = '0;
      a_wdata = '0;
    end
  end

  // =========================================================================
  // Read Interface — port B, combinational address, 1-cycle data latency
  // =========================================================================
  assign b_we    = 1'b0;
  assign b_wdata = '0;
  assign b_addr  = ADDR_W'({1'b0, rd_pos} * D_MODEL + {1'b0, rd_dim});
  assign rd_data = data_t'(b_rdata);

endmodule
