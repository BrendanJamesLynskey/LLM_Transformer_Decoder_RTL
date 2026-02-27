// =============================================================================
// bram_sp.sv - Generic Single-Port Block RAM
// =============================================================================
// Parameterised single-port synchronous BRAM with optional initialisation
// from a hex file. Synthesises to FPGA BRAM primitives (Xilinx, Intel).
//
// Features:
//   - Configurable data width and depth
//   - Single read/write port (read-first behaviour)
//   - Optional $readmemh initialisation for loading weights at synthesis
//   - Write-enable per port
//
// Usage for weights (read-only at inference):
//   - Initialise via INIT_FILE with pretrained weight values
//   - Drive we=0 during inference, supply addr to read one row per cycle
//
// Usage for general storage:
//   - Drive we/addr/wdata for writes, read data appears next cycle
// =============================================================================

module bram_sp
  import transformer_pkg::*;
#(
  parameter int DATA_WIDTH = 16,      // Bit width of each word
  parameter int DEPTH      = 1024,    // Number of words
  parameter     INIT_FILE  = ""       // Optional hex init file path ("" = no init)
)(
  input  logic                       clk,
  input  logic                       we,       // Write enable
  input  logic [$clog2(DEPTH)-1:0]   addr,     // Read/write address
  input  logic [DATA_WIDTH-1:0]      wdata,    // Write data
  output logic [DATA_WIDTH-1:0]      rdata     // Read data (1-cycle latency)
);

  // Memory array
  logic [DATA_WIDTH-1:0] mem [DEPTH];

  // Optional initialisation from hex file
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, mem);
    end
  end

  // Synchronous read/write (read-first)
  always_ff @(posedge clk) begin
    rdata <= mem[addr];
    if (we)
      mem[addr] <= wdata;
  end

endmodule
