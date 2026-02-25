# LLM Transformer Decoder RTL Accelerator

A synthesizable SystemVerilog implementation of a **Transformer Decoder block** optimized for LLM inference, complete with SystemVerilog and CocoTB testbenches.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                  Transformer Decoder Block                │
│                                                          │
│  token_emb ──► [LayerNorm 1] ──► [Multi-Head Attention]  │
│       │                                  │               │
│       └──────────── (+) ◄────────────────┘  Residual 1   │
│                      │                                   │
│                      ▼                                   │
│              [LayerNorm 2] ──► [Feed-Forward Network]    │
│                      │                   │               │
│                      └──── (+) ◄─────────┘  Residual 2   │
│                             │                            │
│                             ▼                            │
│                          out_emb                         │
└──────────────────────────────────────────────────────────┘
```

This is a **pre-norm** decoder architecture (GPT-2/LLaMA style) implementing autoregressive inference with KV-cache support.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_MODEL` | 64 | Embedding dimension |
| `N_HEADS` | 4 | Attention heads |
| `D_HEAD` | 16 | Per-head dimension |
| `D_FF` | 256 | FFN inner dimension (4× model) |
| `MAX_SEQ_LEN` | 128 | Maximum sequence length |
| `DATA_WIDTH` | 16 | Fixed-point width (Q8.8) |

## Module Hierarchy

```
transformer_decoder          (Top-level decoder block)
├── layer_norm               (Pre-attention & pre-FFN normalization)
├── multi_head_attention     (Causal multi-head self-attention)
│   └── [uses KV-cache]
├── feed_forward             (Two-layer FFN with ReLU)
├── softmax_unit             (PWL-approximate softmax)
├── systolic_array           (Matrix multiply engine)
│   └── processing_element   (Single MAC unit)
└── transformer_pkg          (Parameters, types, FP utilities)
```

## Fixed-Point Arithmetic

All computation uses **Q8.8 signed fixed-point** (16-bit) with 32-bit accumulators for MAC operations:

- **Range**: −128.0 to +127.996 (resolution: 1/256 ≈ 0.0039)
- **MAC**: Full-precision 32-bit accumulation, truncated on output
- **Softmax**: Piecewise-linear exponential approximation
- **LayerNorm**: LUT-based reciprocal square root

## Project Structure

```
├── rtl/
│   ├── transformer_pkg.sv        # Package: parameters, types, FP functions
│   ├── processing_element.sv     # Systolic PE (MAC unit)
│   ├── systolic_array.sv         # NxN systolic matrix multiply
│   ├── softmax_unit.sv           # PWL softmax approximation
│   ├── layer_norm.sv             # Layer normalization
│   ├── multi_head_attention.sv   # Multi-head causal attention + KV cache
│   ├── feed_forward.sv           # Position-wise FFN (ReLU)
│   └── transformer_decoder.sv    # Top-level decoder block
├── tb/
│   ├── sv/                       # SystemVerilog testbenches
│   │   ├── tb_processing_element.sv
│   │   ├── tb_systolic_array.sv
│   │   ├── tb_softmax.sv
│   │   └── tb_transformer_decoder.sv
│   └── cocotb/                   # Python CocoTB testbenches
│       ├── test_processing_element.py
│       ├── test_softmax.py
│       ├── Makefile.pe
│       └── Makefile.softmax
├── scripts/
│   ├── run_sim.sh                # Master simulation runner (iverilog)
│   ├── verify_behavioral.py      # Bit-accurate behavioral verification (50 tests)
│   └── lint_check.py             # RTL structural lint checker
├── docs/
│   ├── report.md                 # Technical report (Markdown)
│   └── report.pdf                # Technical report (PDF)
└── README.md
```

## Verification

### Quick Verification (no simulator needed)

The behavioral verification suite mirrors the RTL at the bit level using identical Q8.8 arithmetic. It requires only Python 3.8+:

```bash
python3 scripts/verify_behavioral.py
```

This runs **50 tests** across all modules with golden-model comparison:

| Module | Tests | Key Checks |
|--------|-------|------------|
| Fixed-Point Utilities | 16 | Roundtrip, multiply, saturating add |
| Processing Element | 9 | MAC, forwarding, clear, 20-op random golden model |
| Systolic Array | 8 | Single element, 2×2 matmul `[[19,22],[43,50]]`, clear |
| Softmax Unit | 8 | Uniform, dominant, ordering, sum≈1.0, back-to-back |
| Layer Normalization | 5 | Constant→zero, symmetry, gamma/beta, centering |
| Feed-Forward Network | 4 | ReLU zeroing, bias propagation |
| Decoder Integration | 7 | Full pipeline, KV-cache, 2-token sequential |

### RTL Lint

```bash
python3 scripts/lint_check.py
```

Validates module/endmodule balance, package imports, reset patterns, and cross-file instantiation resolution across all 8 RTL files.

### RTL Simulation (requires iverilog)

#### Prerequisites

- **Python** ≥ 3.8 (for behavioral verification — no other dependencies)
- **Icarus Verilog** ≥ 12.0 (for RTL simulation, SystemVerilog 2012 support)
- **CocoTB** ≥ 1.8 (for Python-driven RTL testbenches)

```bash
# Ubuntu/Debian
sudo apt-get install iverilog
pip install cocotb

# macOS
brew install icarus-verilog
pip install cocotb
```

```bash
# Run all tests
./scripts/run_sim.sh all

# Run individual tests
./scripts/run_sim.sh pe         # Processing Element
./scripts/run_sim.sh systolic   # Systolic Array
./scripts/run_sim.sh softmax    # Softmax Unit
./scripts/run_sim.sh decoder    # Full Decoder (integration)
```

### Running CocoTB Tests

```bash
cd tb/cocotb

# Processing Element tests
make -f Makefile.pe

# Softmax tests
make -f Makefile.softmax
```

## Design Decisions

### Why Q8.8 Fixed-Point?
Integer-only datapaths eliminate the need for floating-point units, drastically reducing area and power. Q8.8 provides sufficient dynamic range for inference in small models while keeping multipliers at 16×16 bits—a sweet spot for FPGA DSP48 blocks.

### Why Systolic Array?
The systolic architecture maximizes data reuse: each operand is used N times as it flows through the array, achieving O(N²) compute with O(N) I/O bandwidth. This directly maps to the matrix multiplications dominating transformer inference.

### Why Pre-Norm?
Pre-norm (LayerNorm before attention/FFN) is more stable for training and produces identical results at inference time. It matches the architecture used by GPT-2, LLaMA, and most modern LLMs.

### KV-Cache Strategy
During autoregressive generation, only the current token's Q is computed fresh; K and V from all prior positions are cached. This reduces per-token compute from O(n²·d) to O(n·d) for attention.

## Extending the Design

**Scaling up**: Increase `D_MODEL`, `N_HEADS`, `D_FF` in `transformer_pkg.sv`. The systolic array dimensions (`PE_ROWS`, `PE_COLS`) control compute throughput.

**Multi-layer**: Instantiate N `transformer_decoder` blocks with a sequencer FSM that chains them. Weight memories can be shared (time-multiplexed) or replicated.

**FPGA targeting**: The design is synthesizable as-is for Xilinx/Intel FPGAs. Replace the combinational weight arrays with BRAM interfaces for practical implementations.

## License

MIT License. See `LICENSE` for details.
