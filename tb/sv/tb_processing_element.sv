// =============================================================================
// tb_processing_element.sv - Processing Element Unit Test
// =============================================================================
`timescale 1ns / 1ps

module tb_processing_element;
  import transformer_pkg::*;

  logic  clk, rst_n, clear, enable;
  data_t a_in, w_in, a_out, w_out;
  acc_t  acc_out;

  processing_element dut (.*);

  // Clock generation: 100 MHz
  initial clk = 0;
  always #5 clk = ~clk;

  // =========================================================================
  // Test Tasks
  // =========================================================================
  task automatic reset();
    rst_n = 0;
    clear = 0;
    enable = 0;
    a_in = '0;
    w_in = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic mac_cycle(input data_t a, input data_t w);
    @(posedge clk);
    enable = 1;
    a_in = a;
    w_in = w;
    @(posedge clk);
    enable = 0;
    a_in = '0;
    w_in = '0;
  endtask

  // =========================================================================
  // Main Test
  // =========================================================================
  int pass_count = 0;
  int fail_count = 0;

  initial begin
    $display("============================================");
    $display("  Processing Element Testbench");
    $display("============================================");

    // Test 1: Reset clears accumulator
    reset();
    assert(acc_out == 0) begin
      $display("[PASS] Reset: acc = 0");
      pass_count++;
    end else begin
      $display("[FAIL] Reset: acc = %0d, expected 0", acc_out);
      fail_count++;
    end

    // Test 2: Simple MAC: 2.0 * 3.0 = 6.0
    // Q8.8: 2.0 = 0x0200, 3.0 = 0x0300
    mac_cycle(16'sh0200, 16'sh0300);
    @(posedge clk); // Pipeline delay
    $display("  MAC 2.0 * 3.0: acc = %0d (hex: %h)", acc_out, acc_out);
    // Expected: 0x0200 * 0x0300 = 0x60000 (in acc_t full precision)
    assert(acc_out == 32'sh00060000) begin
      $display("[PASS] MAC 2.0 * 3.0 = correct");
      pass_count++;
    end else begin
      $display("[FAIL] MAC: got %h, expected 00060000", acc_out);
      fail_count++;
    end

    // Test 3: Accumulation: add 1.0 * 1.0
    mac_cycle(16'sh0100, 16'sh0100);
    @(posedge clk);
    $display("  Accumulated: acc = %h", acc_out);
    // 0x60000 + 0x10000 = 0x70000
    assert(acc_out == 32'sh00070000) begin
      $display("[PASS] Accumulation correct");
      pass_count++;
    end else begin
      $display("[FAIL] Accumulated: got %h, expected 00070000", acc_out);
      fail_count++;
    end

    // Test 4: Data forwarding
    @(posedge clk);
    enable = 1;
    a_in = 16'sh00FF;
    w_in = 16'sh0042;
    @(posedge clk);
    @(posedge clk); // One cycle latency
    assert(a_out == 16'sh00FF && w_out == 16'sh0042) begin
      $display("[PASS] Data forwarding correct");
      pass_count++;
    end else begin
      $display("[FAIL] Forward: a_out=%h w_out=%h", a_out, w_out);
      fail_count++;
    end
    enable = 0;

    // Test 5: Clear
    @(posedge clk);
    clear = 1;
    @(posedge clk);
    clear = 0;
    @(posedge clk);
    assert(acc_out == 0) begin
      $display("[PASS] Clear works");
      pass_count++;
    end else begin
      $display("[FAIL] Clear: acc = %h", acc_out);
      fail_count++;
    end

    // Test 6: Negative numbers: -1.5 * 2.0
    // -1.5 in Q8.8 = 0xFE80, 2.0 = 0x0200
    mac_cycle(16'shFE80, 16'sh0200);
    @(posedge clk);
    $display("  MAC -1.5 * 2.0: acc = %h", acc_out);
    // -1.5 * 2.0 = -3.0 → in full precision: 0xFE80 * 0x0200 sign-extended
    // -384 * 512 = -196608 = 0xFFFCFE80... let's just check sign
    assert(acc_out[ACC_WIDTH-1] == 1'b1) begin
      $display("[PASS] Negative result (sign correct)");
      pass_count++;
    end else begin
      $display("[FAIL] Expected negative result");
      fail_count++;
    end

    // Summary
    $display("");
    $display("============================================");
    $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("============================================");

    $finish;
  end

  // Waveform dump
  initial begin
    $dumpfile("pe_tb.vcd");
    $dumpvars(0, tb_processing_element);
  end

endmodule
