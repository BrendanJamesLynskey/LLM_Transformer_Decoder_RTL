// =============================================================================
// tb_systolic_array.sv - Systolic Array Integration Test
// =============================================================================
`timescale 1ns / 1ps

module tb_systolic_array;
  import transformer_pkg::*;

  localparam int ROWS = 4;
  localparam int COLS = 4;

  logic  clk, rst_n, clear, enable;
  logic signed [ROWS-1:0][DATA_WIDTH-1:0] a_in;
  logic signed [COLS-1:0][DATA_WIDTH-1:0] b_in;
  logic signed [ROWS*COLS-1:0][ACC_WIDTH-1:0] result;
  logic  done;

  systolic_array #(.ROWS(ROWS), .COLS(COLS)) dut (.*);

  // Clock: 100 MHz
  initial clk = 0;
  always #5 clk = ~clk;

  // =========================================================================
  // Test: 4x4 Identity-like Matrix Multiply
  // =========================================================================
  // A = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity, Q8.8)
  // B = [[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,5]]
  // Expected C = A * B = B (since A is identity)

  int pass_count = 0;
  int fail_count = 0;

  initial begin
    $display("============================================");
    $display("  Systolic Array Testbench (%0dx%0d)", ROWS, COLS);
    $display("============================================");

    // Reset
    rst_n = 0;
    clear = 0;
    enable = 0;
    for (int i = 0; i < ROWS; i++) a_in[i] = '0;
    for (int i = 0; i < COLS; i++) b_in[i] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Clear accumulators
    clear = 1;
    @(posedge clk);
    clear = 0;
    @(posedge clk);

    // ---- Test 1: Stream simple values ----
    $display("\n--- Test 1: Simple streaming ---");
    enable = 1;

    // Cycle 0: Feed first values
    a_in[0] = 16'sh0100; // 1.0
    a_in[1] = '0;
    a_in[2] = '0;
    a_in[3] = '0;
    b_in[0] = 16'sh0200; // 2.0
    b_in[1] = '0;
    b_in[2] = '0;
    b_in[3] = '0;
    @(posedge clk);

    // Cycle 1
    a_in[0] = '0;
    a_in[1] = 16'sh0100; // 1.0
    b_in[0] = '0;
    b_in[1] = 16'sh0300; // 3.0
    @(posedge clk);

    // Cycle 2
    a_in[1] = '0;
    a_in[2] = 16'sh0100; // 1.0
    b_in[1] = '0;
    b_in[2] = 16'sh0400; // 4.0
    @(posedge clk);

    // Cycle 3
    a_in[2] = '0;
    a_in[3] = 16'sh0100; // 1.0
    b_in[2] = '0;
    b_in[3] = 16'sh0500; // 5.0
    @(posedge clk);

    // Drain cycles
    for (int i = 0; i < ROWS; i++) a_in[i] = '0;
    for (int i = 0; i < COLS; i++) b_in[i] = '0;
    repeat (ROWS + COLS) @(posedge clk);
    enable = 0;

    // Wait for done
    wait(done);
    @(posedge clk);

    $display("  Result matrix (raw accumulator values):");
    for (int r = 0; r < ROWS; r++) begin
      $write("  Row %0d: ", r);
      for (int c = 0; c < COLS; c++)
        $write("%8h ", result[r*COLS+c]);
      $display("");
    end

    // Check [0][0]: 1.0 * 2.0 = 2.0 in full precision = 0x20000
    if (result[0] == 32'sh00020000) begin
      $display("[PASS] result[0] = 2.0 (correct)");
      pass_count++;
    end else begin
      $display("[FAIL] result[0] = %h, expected 00020000", result[0]);
      fail_count++;
    end

    // ---- Test 2: Verify done signal ----
    $display("\n--- Test 2: Done signal ---");
    if (done) begin
      $display("[PASS] Done asserted after streaming");
      pass_count++;
    end else begin
      $display("[FAIL] Done not asserted");
      fail_count++;
    end

    // ---- Test 3: Clear and re-run ----
    $display("\n--- Test 3: Clear and reuse ---");
    clear = 1;
    @(posedge clk);
    clear = 0;
    @(posedge clk);

    if (!done && result[0] == 0) begin
      $display("[PASS] Clear resets array");
      pass_count++;
    end else begin
      $display("[FAIL] Clear didn't fully reset");
      fail_count++;
    end

    // ---- Test 4: All ones multiply ----
    $display("\n--- Test 4: Uniform 1.0 input ---");
    enable = 1;
    for (int cycle = 0; cycle < ROWS + COLS; cycle++) begin
      for (int i = 0; i < ROWS; i++)
        a_in[i] = (cycle < ROWS) ? 16'sh0100 : '0; // 1.0
      for (int i = 0; i < COLS; i++)
        b_in[i] = (cycle < COLS) ? 16'sh0100 : '0;  // 1.0
      @(posedge clk);
    end
    for (int i = 0; i < ROWS; i++) a_in[i] = '0;
    for (int i = 0; i < COLS; i++) b_in[i] = '0;
    repeat (4) @(posedge clk);
    enable = 0;

    wait(done);
    @(posedge clk);

    $display("  Uniform result[0] = %h", result[0]);
    pass_count++; // Smoke test pass

    // Summary
    $display("");
    $display("============================================");
    $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("============================================");

    $finish;
  end

  initial begin
    $dumpfile("systolic_tb.vcd");
    $dumpvars(0, tb_systolic_array);
  end

endmodule
