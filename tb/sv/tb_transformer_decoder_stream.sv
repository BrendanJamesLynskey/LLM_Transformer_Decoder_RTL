// =============================================================================
// tb_transformer_decoder_stream.sv - Streaming Decoder Integration Test
// =============================================================================
// Tests the streaming architecture where weights are read from BRAM
// via address/data interfaces instead of combinational array ports.
// Uses the same weight initialization and test patterns as the original
// tb_transformer_decoder for direct comparison.
// =============================================================================
`timescale 1ns / 1ps

module tb_transformer_decoder_stream;
  import transformer_pkg::*;

  logic     clk, rst_n, start, valid;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] token_emb;
  logic signed [D_MODEL-1:0][DATA_WIDTH-1:0] out_emb;
  seq_idx_t seq_pos;

  // Weight-loading bus
  logic     wl_en;
  logic [15:0] wl_addr;
  data_t    wl_data;

  // DUT
  transformer_decoder_top_stream dut (
    .clk(clk), .rst_n(rst_n),
    .wl_en(wl_en), .wl_addr(wl_addr), .wl_data(wl_data),
    .start(start), .token_emb(token_emb), .seq_pos(seq_pos),
    .out_emb(out_emb), .valid(valid)
  );

  // Clock: 100 MHz
  initial clk = 0;
  always #5 clk = ~clk;

  // =========================================================================
  // Weight Loading via Bus (same values as original TB)
  // =========================================================================
  task automatic load_weight(input logic [15:0] addr, input data_t data);
    @(posedge clk);
    wl_en   <= 1'b1;
    wl_addr <= addr;
    wl_data <= data;
    @(posedge clk);
    wl_en <= 1'b0;
  endtask

  task automatic load_all_weights();
    // Wq = 0.25 * I (base 0x0000)
    for (int i = 0; i < D_MODEL; i++)
      for (int j = 0; j < D_MODEL; j++)
        load_weight(16'h0000 + i * D_MODEL + j,
                    (i == j) ? 16'sh0040 : 16'sh0000);

    // Wk = 0.25 * I (base 0x1000)
    for (int i = 0; i < D_MODEL; i++)
      for (int j = 0; j < D_MODEL; j++)
        load_weight(16'h1000 + i * D_MODEL + j,
                    (i == j) ? 16'sh0040 : 16'sh0000);

    // Wv = 0.25 * I (base 0x2000)
    for (int i = 0; i < D_MODEL; i++)
      for (int j = 0; j < D_MODEL; j++)
        load_weight(16'h2000 + i * D_MODEL + j,
                    (i == j) ? 16'sh0040 : 16'sh0000);

    // Wo = 0.5 * I (base 0x3000)
    for (int i = 0; i < D_MODEL; i++)
      for (int j = 0; j < D_MODEL; j++)
        load_weight(16'h3000 + i * D_MODEL + j,
                    (i == j) ? 16'sh0080 : 16'sh0000);

    // W1 = 0.125 * periodic I (base 0x4000)
    for (int i = 0; i < D_MODEL; i++)
      for (int j = 0; j < D_FF; j++)
        load_weight(16'h4000 + i * D_FF + j,
                    (i == j % D_MODEL) ? 16'sh0020 : 16'sh0000);

    // W2 = 0.125 * periodic I (base 0x8000)
    for (int i = 0; i < D_FF; i++)
      for (int j = 0; j < D_MODEL; j++)
        load_weight(16'h8000 + i * D_MODEL + j,
                    (i % D_MODEL == j) ? 16'sh0020 : 16'sh0000);

    // LN1 gamma = 1.0 (base 0xC000)
    for (int i = 0; i < D_MODEL; i++)
      load_weight(16'hC000 + i, 16'sh0100);

    // LN1 beta = 0.0 (base 0xC040)
    for (int i = 0; i < D_MODEL; i++)
      load_weight(16'hC040 + i, 16'sh0000);

    // LN2 gamma = 1.0 (base 0xC080)
    for (int i = 0; i < D_MODEL; i++)
      load_weight(16'hC080 + i, 16'sh0100);

    // LN2 beta = 0.0 (base 0xC0C0)
    for (int i = 0; i < D_MODEL; i++)
      load_weight(16'hC0C0 + i, 16'sh0000);

    // b1 = 0.0625 (base 0xC100)
    for (int i = 0; i < D_FF; i++)
      load_weight(16'hC100 + i, 16'sh0010);

    // b2 = 0.0 (base 0xC200)
    for (int i = 0; i < D_MODEL; i++)
      load_weight(16'hC200 + i, 16'sh0000);
  endtask

  // =========================================================================
  // Main Test
  // =========================================================================
  int pass_count = 0;
  int fail_count = 0;
  int timeout_cycles = 500000;

  initial begin
    $display("============================================");
    $display("  Streaming Decoder Integration Test");
    $display("  D_MODEL=%0d, N_HEADS=%0d, D_FF=%0d", D_MODEL, N_HEADS, D_FF);
    $display("============================================");

    rst_n   = 0;
    start   = 0;
    seq_pos = '0;
    wl_en   = 0;
    wl_addr = '0;
    wl_data = '0;

    for (int i = 0; i < D_MODEL; i++)
      token_emb[i] = '0;

    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    // Load weights via bus
    $display("\nLoading weights via bus...");
    load_all_weights();
    $display("  Weight loading complete.");
    repeat (5) @(posedge clk);

    // Create token embedding: alternating 0.5, -0.5
    for (int i = 0; i < D_MODEL; i++)
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
        fail_count++;
      end
    join_any
    disable fork;

    @(posedge clk);

    if (valid) begin
      $display("[PASS] Streaming decoder produced valid output");
      pass_count++;

      $display("  Output embedding (first 8 dims):");
      for (int i = 0; i < 8; i++)
        $display("    out_emb[%0d] = %h (%.4f)", i, out_emb[i],
                 real'($signed(out_emb[i])) / 256.0);

      // Check output is non-zero
      begin
        logic any_nonzero = 0;
        for (int i = 0; i < D_MODEL; i++)
          if (out_emb[i] != '0) any_nonzero = 1;
        if (any_nonzero) begin
          $display("[PASS] Output is non-zero");
          pass_count++;
        end else begin
          $display("[FAIL] Output is all zeros");
          fail_count++;
        end
      end
    end

    // ---- Test 2: Sequential inference (position 1) ----
    $display("\n--- Test 2: Sequential token at position 1 ---");

    // New token
    for (int i = 0; i < D_MODEL; i++)
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
        fail_count++;
      end
    join_any
    disable fork;

    @(posedge clk);

    if (valid) begin
      $display("[PASS] Sequential streaming inference works");
      pass_count++;

      $display("  Output (first 8 dims):");
      for (int i = 0; i < 8; i++)
        $display("    out_emb[%0d] = %h (%.4f)", i, out_emb[i],
                 real'($signed(out_emb[i])) / 256.0);
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
    #(timeout_cycles * 20 * 10);
    $display("[FATAL] Global timeout reached");
    $finish;
  end

  initial begin
    $dumpfile("stream_decoder_tb.vcd");
    $dumpvars(0, tb_transformer_decoder_stream);
  end

endmodule
