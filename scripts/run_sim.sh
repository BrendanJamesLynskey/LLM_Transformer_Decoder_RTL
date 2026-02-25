#!/bin/bash
# =============================================================================
# run_sim.sh - Master Simulation Script
# =============================================================================
# Runs all SystemVerilog testbenches using Icarus Verilog (iverilog).
# Usage: ./scripts/run_sim.sh [pe|systolic|softmax|decoder|all]
# =============================================================================

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RTL_DIR="$PROJ_ROOT/rtl"
TB_DIR="$PROJ_ROOT/tb/sv"
SIM_DIR="$PROJ_ROOT/sim_results"

mkdir -p "$SIM_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# Common RTL files
RTL_PKG="$RTL_DIR/transformer_pkg.sv"

run_pe() {
    echo -e "${CYAN}========== Processing Element Test ==========${NC}"
    iverilog -g2012 -o "$SIM_DIR/pe_tb" \
        "$RTL_PKG" \
        "$RTL_DIR/processing_element.sv" \
        "$TB_DIR/tb_processing_element.sv"
    vvp "$SIM_DIR/pe_tb" | tee "$SIM_DIR/pe_results.log"
    echo ""
}

run_systolic() {
    echo -e "${CYAN}========== Systolic Array Test ==========${NC}"
    iverilog -g2012 -o "$SIM_DIR/systolic_tb" \
        "$RTL_PKG" \
        "$RTL_DIR/processing_element.sv" \
        "$RTL_DIR/systolic_array.sv" \
        "$TB_DIR/tb_systolic_array.sv"
    vvp "$SIM_DIR/systolic_tb" | tee "$SIM_DIR/systolic_results.log"
    echo ""
}

run_softmax() {
    echo -e "${CYAN}========== Softmax Unit Test ==========${NC}"
    iverilog -g2012 -o "$SIM_DIR/softmax_tb" \
        "$RTL_PKG" \
        "$RTL_DIR/softmax_unit.sv" \
        "$TB_DIR/tb_softmax.sv"
    vvp "$SIM_DIR/softmax_tb" | tee "$SIM_DIR/softmax_results.log"
    echo ""
}

run_decoder() {
    echo -e "${CYAN}========== Transformer Decoder Test ==========${NC}"
    iverilog -g2012 -o "$SIM_DIR/decoder_tb" \
        "$RTL_PKG" \
        "$RTL_DIR/processing_element.sv" \
        "$RTL_DIR/systolic_array.sv" \
        "$RTL_DIR/softmax_unit.sv" \
        "$RTL_DIR/layer_norm.sv" \
        "$RTL_DIR/multi_head_attention.sv" \
        "$RTL_DIR/feed_forward.sv" \
        "$RTL_DIR/transformer_decoder.sv" \
        "$TB_DIR/tb_transformer_decoder.sv"
    vvp "$SIM_DIR/decoder_tb" | tee "$SIM_DIR/decoder_results.log"
    echo ""
}

case "${1:-all}" in
    pe)       run_pe ;;
    systolic) run_systolic ;;
    softmax)  run_softmax ;;
    decoder)  run_decoder ;;
    all)
        run_pe
        run_systolic
        run_softmax
        run_decoder
        echo -e "${GREEN}========== All Tests Complete ==========${NC}"
        ;;
    *)
        echo "Usage: $0 [pe|systolic|softmax|decoder|all]"
        exit 1
        ;;
esac
