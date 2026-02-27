// =============================================================================
// bram_dp.sv - Generic True Dual-Port Block RAM
// =============================================================================
// Parameterised true dual-port synchronous BRAM with optional initialisation.
// Both ports can independently read or write on every clock cycle.
// Synthesises to FPGA true dual-port BRAM primitives.
//
// Features:
//   - Configurable data width and depth
//   - Two fully independent read/write ports (A and B)
//   - Read-first behaviour on both ports
//   - Optional $readmemh initialisation
//   - No read/write collision handling (undefined if both ports write
//     to the same address simultaneously)
//
// Primary use: KV-Cache
//   - Port A: Write new K/V vectors at the current sequence position
//   - Port B: Read cached K/V vectors during attention score computation
//
// Memory layout for KV-Cache:
//   Each address stores one DATA_WIDTH-bit element.
//   A full D_MODEL-wide vector at position t occupies addresses
//   [t * D_MODEL .. (t+1) * D_MODEL - 1].
// =============================================================================

module bram_dp
  import transformer_pkg::*;
#(
  parameter int DATA_WIDTH = 16,      // Bit width of each word
  parameter int DEPTH      = 1024,    // Number of words
  parameter     INIT_FILE  = ""       // Optional hex init file path ("" = no init)
)(
  input  logic                       clk,

  // Port A (typically write port for new K/V)
  input  logic                       a_we,
  input  logic [$clog2(DEPTH)-1:0]   a_addr,
  input  logic [DATA_WIDTH-1:0]      a_wdata,
  output logic [DATA_WIDTH-1:0]      a_rdata,

  // Port B (typically read port for cached values)
  input  logic                       b_we,
  input  logic [$clog2(DEPTH)-1:0]   b_addr,
  input  logic [DATA_WIDTH-1:0]      b_wdata,
  output logic [DATA_WIDTH-1:0]      b_rdata
);

  // Memory array
  logic [DATA_WIDTH-1:0] mem [DEPTH];

  // Optional initialisation from hex file
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, mem);
    end
  end

  // Port A: synchronous read/write (read-first)
  always_ff @(posedge clk) begin
    a_rdata <= mem[a_addr];
    if (a_we)
      mem[a_addr] <= a_wdata;
  end

  // Port B: synchronous read/write (read-first)
  always_ff @(posedge clk) begin
    b_rdata <= mem[b_addr];
    if (b_we)
      mem[b_addr] <= b_wdata;
  end

endmodule
