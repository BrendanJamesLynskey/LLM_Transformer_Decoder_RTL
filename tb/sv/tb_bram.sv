// =============================================================================
// tb_bram.sv - Testbench for bram_sp and bram_dp modules
// =============================================================================
`timescale 1ns / 1ps

module tb_bram;
  import transformer_pkg::*;

  logic clk;
  initial clk = 0;
  always #5 clk = ~clk;

  int pass_count = 0;
  int fail_count = 0;

  // =========================================================================
  // Single-port BRAM (no init)
  // =========================================================================
  logic        sp_we;
  logic [3:0]  sp_addr;
  logic [15:0] sp_wdata, sp_rdata;

  bram_sp #(.DATA_WIDTH(16), .DEPTH(16), .INIT_FILE("")) u_sp (
    .clk(clk), .we(sp_we), .addr(sp_addr), .wdata(sp_wdata), .rdata(sp_rdata)
  );

  // =========================================================================
  // Dual-port BRAM (no init)
  // =========================================================================
  logic        dp_a_we, dp_b_we;
  logic [3:0]  dp_a_addr, dp_b_addr;
  logic [15:0] dp_a_wdata, dp_b_wdata, dp_a_rdata, dp_b_rdata;

  bram_dp #(.DATA_WIDTH(16), .DEPTH(16), .INIT_FILE("")) u_dp (
    .clk(clk),
    .a_we(dp_a_we), .a_addr(dp_a_addr), .a_wdata(dp_a_wdata), .a_rdata(dp_a_rdata),
    .b_we(dp_b_we), .b_addr(dp_b_addr), .b_wdata(dp_b_wdata), .b_rdata(dp_b_rdata)
  );

  // =========================================================================
  // Helper tasks — drive on negedge, sample on posedge (avoids race)
  // =========================================================================

  task automatic sp_write(input logic [3:0] a, input logic [15:0] d);
    @(negedge clk);
    sp_we = 1; sp_addr = a; sp_wdata = d;
    @(negedge clk);
    sp_we = 0;
  endtask

  task automatic sp_read(input logic [3:0] a, output logic [15:0] d);
    @(negedge clk);
    sp_addr = a; sp_we = 0;
    @(posedge clk);  // BRAM registers addr on this edge
    @(posedge clk);  // rdata valid after this edge
    d = sp_rdata;
  endtask

  task automatic dp_a_write(input logic [3:0] a, input logic [15:0] d);
    @(negedge clk);
    dp_a_we = 1; dp_a_addr = a; dp_a_wdata = d;
    @(negedge clk);
    dp_a_we = 0;
  endtask

  task automatic dp_b_write(input logic [3:0] a, input logic [15:0] d);
    @(negedge clk);
    dp_b_we = 1; dp_b_addr = a; dp_b_wdata = d;
    @(negedge clk);
    dp_b_we = 0;
  endtask

  task automatic dp_read_both(
    input logic [3:0] aa, output logic [15:0] ad,
    input logic [3:0] ba, output logic [15:0] bd
  );
    @(negedge clk);
    dp_a_we = 0; dp_a_addr = aa;
    dp_b_we = 0; dp_b_addr = ba;
    @(posedge clk);
    @(posedge clk);
    ad = dp_a_rdata;
    bd = dp_b_rdata;
  endtask

  // =========================================================================
  // Main test sequence
  // =========================================================================
  initial begin
    logic [15:0] rd_val, rd_a, rd_b;

    $display("============================================");
    $display("  BRAM Module Testbench");
    $display("============================================");

    sp_we = 0; sp_addr = 0; sp_wdata = 0;
    dp_a_we = 0; dp_a_addr = 0; dp_a_wdata = 0;
    dp_b_we = 0; dp_b_addr = 0; dp_b_wdata = 0;
    repeat (4) @(posedge clk);

    // ================================================================
    // Test 1: SP single write/read
    // ================================================================
    $display("\n--- Test 1: SP single write/read ---");
    sp_write(4'd3, 16'hCAFE);
    sp_read(4'd3, rd_val);
    if (rd_val === 16'hCAFE) begin
      $display("[PASS] SP write/read: addr=3 data=0x%h", rd_val);
      pass_count++;
    end else begin
      $display("[FAIL] SP write/read: expected CAFE, got %h", rd_val);
      fail_count++;
    end

    // ================================================================
    // Test 2: SP sequential write then readback (8 values)
    // ================================================================
    $display("\n--- Test 2: SP sequential write/read ---");
    for (int i = 0; i < 8; i++)
      sp_write(i[3:0], 16'hA000 + i[15:0]);

    begin
      int ok = 1;
      for (int i = 0; i < 8; i++) begin
        sp_read(i[3:0], rd_val);
        if (rd_val !== (16'hA000 + i[15:0])) begin
          $display("  [FAIL] addr=%0d expected %h got %h", i, 16'hA000+i[15:0], rd_val);
          ok = 0;
        end
      end
      if (ok) begin
        $display("[PASS] SP sequential: 8 values match");
        pass_count++;
      end else
        fail_count++;
    end

    // ================================================================
    // Test 3: SP init via direct mem load ($readmemh simulation)
    // ================================================================
    $display("\n--- Test 3: SP hex init (simulated) ---");
    for (int i = 0; i < 16; i++)
      u_sp.mem[i] = 16'h0100 + i[15:0];

    sp_read(4'd0, rd_val);
    if (rd_val === 16'h0100) begin
      $display("[PASS] Init addr=0 → 0x%h", rd_val);
      pass_count++;
    end else begin
      $display("[FAIL] Init addr=0 expected 0100, got %h", rd_val);
      fail_count++;
    end

    sp_read(4'd15, rd_val);
    if (rd_val === 16'h010F) begin
      $display("[PASS] Init addr=15 → 0x%h", rd_val);
      pass_count++;
    end else begin
      $display("[FAIL] Init addr=15 expected 010F, got %h", rd_val);
      fail_count++;
    end

    // ================================================================
    // Test 4: DP write A, read B (and vice versa)
    // ================================================================
    $display("\n--- Test 4: Dual-port cross write/read ---");
    dp_a_write(4'd7, 16'hBEEF);
    dp_b_write(4'd9, 16'hDEAD);

    dp_read_both(4'd9, rd_a, 4'd7, rd_b);
    if (rd_a === 16'hDEAD) begin
      $display("[PASS] DP: A reads B's write (addr=9) → 0x%h", rd_a);
      pass_count++;
    end else begin
      $display("[FAIL] DP: A reads addr=9 expected DEAD, got %h", rd_a);
      fail_count++;
    end
    if (rd_b === 16'hBEEF) begin
      $display("[PASS] DP: B reads A's write (addr=7) → 0x%h", rd_b);
      pass_count++;
    end else begin
      $display("[FAIL] DP: B reads addr=7 expected BEEF, got %h", rd_b);
      fail_count++;
    end

    // ================================================================
    // Test 5: DP simultaneous reads from distinct addresses
    // ================================================================
    $display("\n--- Test 5: DP simultaneous reads ---");
    dp_read_both(4'd7, rd_a, 4'd9, rd_b);
    if (rd_a === 16'hBEEF && rd_b === 16'hDEAD) begin
      $display("[PASS] DP simultaneous: A=0x%h B=0x%h", rd_a, rd_b);
      pass_count++;
    end else begin
      $display("[FAIL] DP simultaneous: A=%h (exp BEEF) B=%h (exp DEAD)", rd_a, rd_b);
      fail_count++;
    end

    // ================================================================
    // Test 6: DP init via direct mem load
    // ================================================================
    $display("\n--- Test 6: DP hex init (simulated) ---");
    for (int i = 0; i < 16; i++)
      u_dp.mem[i] = 16'hF000 + i[15:0];

    dp_read_both(4'd0, rd_a, 4'd15, rd_b);
    if (rd_a === 16'hF000 && rd_b === 16'hF00F) begin
      $display("[PASS] DP init: A[0]=0x%h B[15]=0x%h", rd_a, rd_b);
      pass_count++;
    end else begin
      $display("[FAIL] DP init: A[0]=%h (exp F000) B[15]=%h (exp F00F)", rd_a, rd_b);
      fail_count++;
    end

    $display("");
    $display("============================================");
    $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("============================================");
    $finish;
  end

  initial begin
    $dumpfile("bram_tb.vcd");
    $dumpvars(0, tb_bram);
  end

endmodule
