// =============================================================================
// weight_bram.sv - 2D Weight Matrix Stored in Flat BRAM
// =============================================================================
// Stores a ROWS × COLS weight matrix in a single-port BRAM of depth
// ROWS * COLS. Provides:
//
//   - Write port: bulk-load weights one element at a time (row-major order).
//     Assert wr_en and supply flat_addr + wr_data.
//
//   - Column-read interface: to read column `col`, assert col_rd_start with
//     the target col_rd_idx. The module sequentially reads ROWS elements
//     from BRAM (addresses col, col+COLS, col+2*COLS, ...) into a registered
//     output buffer col_data[0..ROWS-1]. When complete, col_rd_done pulses.
//
// The column-read interface matches the access pattern in the compute
// modules (S_PROJ_QKV, S_LINEAR1, etc.) which need one full column per
// output dimension.
//
// Initialisation: pass INIT_FILE to pre-load the BRAM at synthesis/sim.
//   File format: row-major hex, one 16-bit value per line.
//   Address mapping: mem[r * COLS + c] = weight[r][c]
// =============================================================================

module weight_bram
  import transformer_pkg::*;
#(
  parameter int ROWS      = D_MODEL,   // First dimension (e.g. D_MODEL)
  parameter int COLS      = D_MODEL,   // Second dimension (e.g. D_MODEL or D_FF)
  parameter     INIT_FILE = ""         // Optional hex init file
)(
  input  logic                              clk,
  input  logic                              rst_n,

  // --- Bulk write interface (for initialisation) ---
  input  logic                              wr_en,
  input  logic [$clog2(ROWS*COLS)-1:0]      wr_addr,   // Flat address (row-major)
  input  data_t                             wr_data,

  // --- Column-read interface ---
  input  logic                              col_rd_start,
  input  logic [$clog2(COLS)-1:0]           col_rd_idx,   // Which column to read
  output data_t                             col_data [ROWS], // Registered output buffer
  output logic                              col_rd_done
);

  localparam int TOTAL = ROWS * COLS;
  localparam int ADDR_W = $clog2(TOTAL);

  // BRAM interface signals
  logic                  bram_we;
  logic [ADDR_W-1:0]     bram_addr;
  logic [DATA_WIDTH-1:0] bram_wdata;
  logic [DATA_WIDTH-1:0] bram_rdata;

  // Underlying BRAM
  bram_sp #(
    .DATA_WIDTH (DATA_WIDTH),
    .DEPTH      (TOTAL),
    .INIT_FILE  (INIT_FILE)
  ) u_bram (
    .clk   (clk),
    .we    (bram_we),
    .addr  (bram_addr),
    .wdata (bram_wdata),
    .rdata (bram_rdata)
  );

  // Column-read FSM
  typedef enum logic [1:0] {
    RD_IDLE,
    RD_FETCH,    // Issue read addresses to BRAM
    RD_CAPTURE,  // Capture the last read (1-cycle pipeline delay)
    RD_DONE
  } rd_state_t;

  rd_state_t rd_state, rd_state_next;

  logic [$clog2(ROWS):0] rd_row;       // Current row being read
  logic [$clog2(COLS)-1:0] rd_col_reg; // Latched column index
  logic [$clog2(ROWS):0] capture_row;  // Which row's data is arriving from BRAM

  // =========================================================================
  // Sequential Logic
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_state    <= RD_IDLE;
      rd_row      <= '0;
      capture_row <= '0;
      rd_col_reg  <= '0;
      col_rd_done <= 1'b0;
    end else begin
      rd_state    <= rd_state_next;
      col_rd_done <= 1'b0;

      case (rd_state)
        RD_IDLE: begin
          if (col_rd_start) begin
            rd_col_reg  <= col_rd_idx;
            rd_row      <= '0;
            capture_row <= '0;
          end
        end

        RD_FETCH: begin
          // Issue read for row rd_row, column rd_col_reg
          // Capture arrives next cycle
          if (rd_row < ROWS[$clog2(ROWS):0]) begin
            rd_row <= rd_row + 1;
          end
          // Capture the data that was read 1 cycle ago
          if (capture_row < rd_row && capture_row < ROWS[$clog2(ROWS):0]) begin
            col_data[capture_row] <= data_t'(bram_rdata);
            capture_row <= capture_row + 1;
          end
        end

        RD_CAPTURE: begin
          // Capture final element(s) still in the pipeline
          if (capture_row < ROWS[$clog2(ROWS):0]) begin
            col_data[capture_row] <= data_t'(bram_rdata);
            capture_row <= capture_row + 1;
          end else begin
            col_rd_done <= 1'b1;
          end
        end

        RD_DONE: ;
      endcase
    end
  end

  // =========================================================================
  // Next-State Logic
  // =========================================================================
  always_comb begin
    rd_state_next = rd_state;
    case (rd_state)
      RD_IDLE:    if (col_rd_start) rd_state_next = RD_FETCH;
      RD_FETCH:   if (rd_row >= ROWS[$clog2(ROWS):0]) rd_state_next = RD_CAPTURE;
      RD_CAPTURE: if (capture_row >= ROWS[$clog2(ROWS):0]) rd_state_next = RD_DONE;
      RD_DONE:    rd_state_next = RD_IDLE;
      default:    rd_state_next = RD_IDLE;
    endcase
  end

  // =========================================================================
  // BRAM Address and Write Mux
  // =========================================================================
  always_comb begin
    if (wr_en) begin
      // Write path: external bulk load
      bram_we    = 1'b1;
      bram_addr  = wr_addr;
      bram_wdata = wr_data;
    end else if (rd_state == RD_FETCH && rd_row < ROWS[$clog2(ROWS):0]) begin
      // Read path: sequential column read
      // Address = rd_row * COLS + rd_col_reg
      bram_we    = 1'b0;
      bram_addr  = ADDR_W'(rd_row * COLS + {1'b0, rd_col_reg});
      bram_wdata = '0;
    end else begin
      bram_we    = 1'b0;
      bram_addr  = '0;
      bram_wdata = '0;
    end
  end

endmodule
