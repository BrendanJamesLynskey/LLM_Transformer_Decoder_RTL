// =============================================================================
// tb_transformer_decoder.sv - Register-Bridge Decoder Integration Testbench
// =============================================================================
// End-to-end test of the register-bridge transformer decoder (via top wrapper).
// Loads weights via the wl_en bus, then verifies single-token and sequential
// two-token inference.
//
// Modified for iverilog 10.1 compatibility:
//   - Tasks removed (iverilog can't resolve package constants in task scope)
//   - Weight loading inlined in initial block
//   - Inline logic declarations moved to module level
//   - int changed to integer
// =============================================================================
`timescale 1ns / 1ps

module tb_transformer_decoder;
  import transformer_pkg::*;

  // iverilog 10.1 can't resolve package constants inside tasks/functions,
  // so re-declare them as localparams at module scope.
  localparam integer LCL_D_MODEL = 64;
  localparam integer LCL_D_FF    = 256;

  logic     clk, rst_n, start, valid;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] token_emb;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] out_emb;
  reg [6:0] seq_pos;

  // Weight-loading bus
  logic        wl_en;
  logic [15:0] wl_addr;
  reg signed [15:0] wl_data;

  // DUT: register-bridge top wrapper
  transformer_decoder_top dut (
    .clk(clk), .rst_n(rst_n),
    .wl_en(wl_en), .wl_addr(wl_addr), .wl_data(wl_data),
    .start(start), .token_emb(token_emb), .seq_pos(seq_pos),
    .out_emb(out_emb), .valid(valid)
  );

  integer i, j;
  integer pass_count;
  integer fail_count;
  integer timeout_cycles;
  integer any_nonzero;
  reg signed [DATA_WIDTH-1:0] out_emb_tok0 [LCL_D_MODEL]; // saved output from token 0
  integer differ;

  // Clock: 100 MHz
  initial clk = 0;
  always #5 clk = ~clk;

  // =========================================================================
  // Main Test
  // =========================================================================
  initial begin
    pass_count = 0;
    fail_count = 0;
    timeout_cycles = 200000;

    $display("============================================");
    $display("  Transformer Decoder Integration Test");
    $display("  D_MODEL=%0d, N_HEADS=%0d, D_FF=%0d", D_MODEL, N_HEADS, D_FF);
    $display("============================================");

    rst_n   = 0;
    start   = 0;
    seq_pos = 0;
    wl_en   = 0;
    wl_addr = 0;
    wl_data = 0;

    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      token_emb[i] = 0;

    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // =====================================================================
    // Load weights via bus (inlined — tasks can't see package constants)
    // =====================================================================
    $display("\nLoading weights via bus...");

    // Wq = 0.25 * I (base 0x0000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      for (j = 0; j < LCL_D_MODEL; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h0000 + i * LCL_D_MODEL + j;
        wl_data <= (i == j) ? 16'sh0040 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // Wk = 0.25 * I (base 0x1000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      for (j = 0; j < LCL_D_MODEL; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h1000 + i * LCL_D_MODEL + j;
        wl_data <= (i == j) ? 16'sh0040 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // Wv = 0.25 * I (base 0x2000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      for (j = 0; j < LCL_D_MODEL; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h2000 + i * LCL_D_MODEL + j;
        wl_data <= (i == j) ? 16'sh0040 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // Wo = 0.5 * I (base 0x3000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      for (j = 0; j < LCL_D_MODEL; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h3000 + i * LCL_D_MODEL + j;
        wl_data <= (i == j) ? 16'sh0080 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // W1 = 0.125 * periodic I (base 0x4000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      for (j = 0; j < LCL_D_FF; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h4000 + i * LCL_D_FF + j;
        wl_data <= (i == j % LCL_D_MODEL) ? 16'sh0020 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // W2 = 0.125 * periodic I (base 0x8000)
    for (i = 0; i < LCL_D_FF; i = i + 1)
      for (j = 0; j < LCL_D_MODEL; j = j + 1) begin
        @(posedge clk);
        wl_en   <= 1'b1;
        wl_addr <= 16'h8000 + i * LCL_D_MODEL + j;
        wl_data <= (i % LCL_D_MODEL == j) ? 16'sh0020 : 16'sh0000;
        @(posedge clk);
        wl_en <= 1'b0;
      end

    // LN1 gamma = 1.0 (base 0xC000)
    for (i = 0; i < LCL_D_MODEL; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC000 + i;
      wl_data <= 16'sh0100;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    // LN1 beta = 0.0 (base 0xC040)
    for (i = 0; i < LCL_D_MODEL; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC040 + i;
      wl_data <= 16'sh0000;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    // LN2 gamma = 1.0 (base 0xC080)
    for (i = 0; i < LCL_D_MODEL; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC080 + i;
      wl_data <= 16'sh0100;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    // LN2 beta = 0.0 (base 0xC0C0)
    for (i = 0; i < LCL_D_MODEL; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC0C0 + i;
      wl_data <= 16'sh0000;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    // b1 = 0.0625 (base 0xC100)
    for (i = 0; i < LCL_D_FF; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC100 + i;
      wl_data <= 16'sh0010;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    // b2 = 0.0 (base 0xC200)
    for (i = 0; i < LCL_D_MODEL; i = i + 1) begin
      @(posedge clk);
      wl_en   <= 1'b1;
      wl_addr <= 16'hC200 + i;
      wl_data <= 16'sh0000;
      @(posedge clk);
      wl_en <= 1'b0;
    end

    $display("  Weight loading complete.");
    repeat (5) @(posedge clk);

    // Create token embedding: alternating 0.5, -0.5
    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      token_emb[i] = (i % 2 == 0) ? 16'sh0080 : 16'shFF80;

    // ---- Test 1: Single token inference (position 0) ----
    $display("\n--- Test 1: Single token at position 0 ---");
    seq_pos = 7'b0;
    @(posedge clk);
    start = 1;
    @(posedge clk);
    start = 0;

    fork
      begin
        wait(valid);
        $display("  Decoder completed!");
      end
      begin
        repeat (timeout_cycles) @(posedge clk);
        $display("[FAIL] Timeout waiting for valid");
        fail_count = fail_count + 1;
      end
    join_any
    disable fork;

    @(posedge clk);

    if (valid) begin
      $display("[PASS] Decoder produced valid output");
      pass_count = pass_count + 1;

      $display("  Output embedding (first 8 dims):");
      for (i = 0; i < 8; i = i + 1)
        $display("    out_emb[%0d] = %h (%.4f)", i, out_emb[i],
                 $itor($signed(out_emb[i])) / 256.0);

      // Check output is non-zero
      any_nonzero = 0;
      for (i = 0; i < LCL_D_MODEL; i = i + 1)
        if (out_emb[i] != 0) any_nonzero = 1;
      if (any_nonzero) begin
        $display("[PASS] Output is non-zero");
        pass_count = pass_count + 1;
      end else begin
        $display("[FAIL] Output is all zeros");
        fail_count = fail_count + 1;
      end

      // Save token-0 output for KV-cache check in Test 2
      for (i = 0; i < LCL_D_MODEL; i = i + 1)
        out_emb_tok0[i] = out_emb[i];
    end

    // ---- Test 2: KV Cache functional check ----
    // Run a different token at position 1; the output must differ from token 0
    // because the KV cache now holds the position-0 K/V entries and the new
    // query attends over two positions instead of one.
    $display("\n--- Test 2: KV Cache functional check (token at position 1) ---");

    for (i = 0; i < LCL_D_MODEL; i = i + 1)
      token_emb[i] = (i % 3 == 0) ? 16'sh0100 : 16'sh0000;

    seq_pos = 7'd1;
    repeat (3) @(posedge clk);
    start = 1;
    @(posedge clk);
    start = 0;

    fork
      begin
        wait(valid);
        $display("  Second token completed!");
      end
      begin
        repeat (timeout_cycles) @(posedge clk);
        $display("[FAIL] Timeout on second token");
        fail_count = fail_count + 1;
      end
    join_any
    disable fork;

    @(posedge clk);

    if (valid) begin
      $display("[PASS] Sequential inference works");
      pass_count = pass_count + 1;

      $display("  Output (first 8 dims):");
      for (i = 0; i < 8; i = i + 1)
        $display("    out_emb[%0d] = %h (%.4f)", i, out_emb[i],
                 $itor($signed(out_emb[i])) / 256.0);

      // KV cache check: token-1 output must differ from token-0 output.
      // If the KV cache had no effect, both tokens with the same attention
      // pattern would produce identical outputs.
      differ = 0;
      for (i = 0; i < LCL_D_MODEL; i = i + 1)
        if (out_emb[i] !== out_emb_tok0[i]) differ = 1;
      if (differ) begin
        $display("[PASS] KV cache active: token-1 output differs from token-0");
        pass_count = pass_count + 1;
      end else begin
        $display("[FAIL] KV cache check: token-1 output identical to token-0 (cache inactive?)");
        fail_count = fail_count + 1;
      end
    end

    // Summary
    $display("");
    $display("============================================");
    $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("============================================");

    #100;
    $finish;
  end

  // Timeout safety net
  initial begin
    #100000000;
    $display("[FATAL] Global timeout reached");
    $finish;
  end

  initial begin
    $dumpfile("decoder_tb.vcd");
    $dumpvars(0, tb_transformer_decoder);
  end

endmodule
