# LLM Transformer Decoder RTL Accelerator

A synthesizable SystemVerilog implementation of a **Transformer Decoder block** optimized for LLM inference, with full softmax integration, KV-cache support, and comprehensive verification.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   Transformer Decoder Block                   │
│                                                              │
│  token_emb ──► [LayerNorm 1] ──► [Multi-Head Attention] ◄───┤
│       │                           │  ┌──────────────┐       │
│       │                           ├──┤ softmax_unit │       │
│       │                           │  └──────────────┘       │
│       │                           │  ┌──────────────┐       │
│       │                           └──┤   KV-Cache   │       │
│       │                              └──────────────┘       │
│       └──────────── (+) ◄────────────────┘  Residual 1      │
│                      │                                       │
│                      ▼                                       │
│              [LayerNorm 2] ──► [Feed-Forward Network]        │
│                      │                   │                   │
│                      └──── (+) ◄─────────┘  Residual 2      │
│                             │                                │
│                             ▼                                │
│                          out_emb                             │
└──────────────────────────────────────────────────────────────┘
```

This is a **pre-norm** decoder architecture (GPT-2/LLaMA style) implementing autoregressive inference with KV-cache support and full piecewise-linear softmax normalisation.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_MODEL` | 64 | Embedding dimension |
| `N_HEADS` | 4 | Attention heads |
| `D_HEAD` | 16 | Per-head dimension (D_MODEL / N_HEADS) |
| `D_FF` | 256 | FFN inner dimension (4× model) |
| `MAX_SEQ_LEN` | 128 | Maximum sequence length |
| `DATA_WIDTH` | 16 | Fixed-point width (Q8.8) |
| `ACC_WIDTH` | 32 | Accumulator width for MAC operations |
| `PE_ROWS` | 4 | Systolic array rows |
| `PE_COLS` | 4 | Systolic array columns |

## Module Hierarchy

```
transformer_decoder              (Top-level decoder block)
├── layer_norm                   (Pre-attention & pre-FFN normalisation)
├── multi_head_attention         (Causal multi-head self-attention)
│   └── softmax_unit             (PWL-approximate softmax, time-multiplexed across heads)
│       └── [reads KV-cache]
├── feed_forward                 (Two-layer FFN with ReLU)
├── systolic_array               (Matrix multiply engine — available for tiled projection)
│   └── processing_element       (Single MAC unit)
└── transformer_pkg              (Parameters, types, FP utilities)
```

## Attention Pipeline

The multi-head attention module implements the complete scaled dot-product attention with integrated softmax:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

The pipeline proceeds through these FSM stages:

| Stage | Operation | Cycles (seq_pos=127) |
|-------|-----------|---------------------|
| `S_PROJ_QKV` | Project input → Q, K, V via weight matrices | 64 |
| `S_WRITE_CACHE` | Store K, V in KV-cache at current position | 1 |
| `S_SCORE` | Compute Q·K^T / √d_k for all heads and positions | 512 |
| `S_SOFTMAX_PREP` | Pad scores with −8.0 for causal mask | 1 per head |
| `S_SOFTMAX_RUN` | Run softmax_unit (find-max, exp, sum, normalise) | ~640 per head |
| `S_SOFTMAX_STORE` | Copy probabilities, advance to next head | 1 per head |
| `S_WEIGHTED_SUM` | Compute probability-weighted sum of V vectors | 4 |
| `S_OUTPUT_PROJ` | Project concatenated heads through Wo | 64 |
| **Total** | | **~3,213** |

A single `softmax_unit` instance (VEC_LEN = MAX_SEQ_LEN) is time-multiplexed across all N_HEADS. Causal masking is achieved by padding future positions with −8.0 in Q8.8, which the PWL exponential maps to near-zero probability.

## Fixed-Point Arithmetic

All computation uses **Q8.8 signed fixed-point** (16-bit) with 32-bit accumulators for MAC operations:

| Property | Value |
|----------|-------|
| Format | 1 sign + 7 integer + 8 fractional bits |
| Range | −128.0 to +127.996 |
| Resolution | 1/256 ≈ 0.0039 (~48 dB SQNR) |
| Multiply | 16×16 → 32-bit product, arithmetic right-shift by 8 |
| Accumulate | Full 32-bit precision, truncated on output |
| Addition | Saturating (clamps to representable range) |
| Softmax exp | 4-segment piecewise-linear approximation over [−8, 0] |
| LayerNorm 1/√x | 4-entry LUT in transformer_pkg |

## Project Structure

```
├── rtl/
│   ├── transformer_pkg.sv        # Package: parameters, types, FP functions
│   ├── processing_element.sv     # Systolic PE (MAC unit)
│   ├── systolic_array.sv         # NxN systolic matrix multiply
│   ├── softmax_unit.sv           # PWL softmax approximation
│   ├── layer_norm.sv             # Layer normalisation
│   ├── multi_head_attention.sv   # Multi-head causal attention + softmax + KV cache
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
│   ├── verify_behavioral.py      # Bit-accurate behavioral verification (54 tests)
│   └── lint_check.py             # RTL structural lint checker
├── docs/
│   ├── report.md                 # Technical report (Markdown)
│   └── report.pdf                # Technical report (PDF)
└── README.md
```

## Verification

The project uses three complementary verification layers: a bit-accurate Python behavioural model, structural RTL lint, and iverilog RTL simulation.

### Quick Verification (no simulator needed)

The behavioural verification suite mirrors the RTL at the bit level using identical Q8.8 arithmetic, accumulator widths, and FSM sequencing. It requires only Python 3.8+:

```bash
python3 scripts/verify_behavioral.py
```

This runs **54 tests** across all modules with golden-model comparison:

| Module | Tests | Key Checks |
|--------|-------|------------|
| Fixed-Point Utilities | 16 | Roundtrip, multiply, saturating add, edge values |
| Processing Element | 9 | MAC, forwarding, clear, 20-op randomised golden model |
| Systolic Array | 8 | Single element, 2×2 matmul `[[19,22],[43,50]]`, clear |
| Softmax Unit | 8 | Uniform, dominant, ordering, sum≈1.0, back-to-back |
| Layer Normalisation | 5 | Constant→zero, symmetry, gamma/beta, centering |
| Feed-Forward Network | 4 | ReLU zeroing, bias propagation |
| Decoder Integration | 12 | Full pipeline with softmax, causal mask, KV-cache, 2-token sequential |

The decoder integration tests verify:
- **Single-position softmax**: probability ≈ 0.90 for the valid position, < 0.02 for masked future positions
- **Multi-position softmax**: probability distributes across both active positions with sum ≈ 0.95
- **Causal mask enforcement**: future position probabilities < 0.01
- **End-to-end signal flow**: output energy (4.91) exceeds input energy (2.00), confirming propagation through all stages

### RTL Lint

```bash
python3 scripts/lint_check.py
```

Validates module/endmodule balance, package imports, reset patterns, and cross-file instantiation resolution across all 8 RTL files. Confirms `softmax_unit` is instantiated within `multi_head_attention`.

### RTL Simulation (requires iverilog)

#### Prerequisites

- **Python** ≥ 3.8 (for behavioural verification — no other dependencies)
- **Icarus Verilog** ≥ 12.0 (for RTL simulation, SystemVerilog 2012 support)
- **CocoTB** ≥ 1.8 (for Python-driven RTL testbenches, optional)

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

#### Manual compilation

```bash
# Full decoder with softmax integration
iverilog -g2012 -o sim_decoder \
  rtl/transformer_pkg.sv \
  rtl/processing_element.sv \
  rtl/systolic_array.sv \
  rtl/softmax_unit.sv \
  rtl/layer_norm.sv \
  rtl/multi_head_attention.sv \
  rtl/feed_forward.sv \
  rtl/transformer_decoder.sv \
  tb/sv/tb_transformer_decoder.sv

vvp sim_decoder
```

### CocoTB Tests

```bash
cd tb/cocotb

# Processing Element tests
make -f Makefile.pe

# Softmax tests
make -f Makefile.softmax
```

### Known iverilog Limitations

Icarus Verilog 12.0 has incomplete support for SystemVerilog unpacked array port propagation. This causes `xxxx` values in simulation output for signals passed through unpacked array ports (e.g. `out_emb`, `result`, `probs`). The FSM control flow and scalar signals (e.g. `valid`, `done`) simulate correctly. The bit-accurate Python behavioural model provides full numerical verification independent of these iverilog limitations.

## Design Decisions

### Why Q8.8 Fixed-Point?
Integer-only datapaths eliminate the need for floating-point units, drastically reducing area and power. Q8.8 provides sufficient dynamic range for inference in small models while keeping multipliers at 16×16 bits — a sweet spot for FPGA DSP48 blocks.

### Why Systolic Array?
The systolic architecture maximises data reuse: each operand is used N times as it flows through the array, achieving O(N²) compute with O(N) I/O bandwidth. This directly maps to the matrix multiplications dominating transformer inference.

### Why Pre-Norm?
Pre-norm (LayerNorm before attention/FFN) is more stable for training and produces identical results at inference time. It matches the architecture used by GPT-2, LLaMA, and most modern LLMs.

### Why Time-Multiplexed Softmax?
A single softmax_unit is shared across all attention heads to minimise area. Each head's scores are padded with −8.0 for masked positions and fed through the unit sequentially. This trades latency (~2,568 cycles for 4 heads) for a 4× reduction in softmax hardware. Parallelising to N_HEADS instances is a straightforward optimisation when latency is the constraint.

### KV-Cache Strategy
During autoregressive generation, only the current token's Q is computed fresh; K and V from all prior positions are cached. This reduces per-token compute from O(n²·d) to O(n·d) for attention.

## Extending the Design

**Scaling up**: Increase `D_MODEL`, `N_HEADS`, `D_FF` in `transformer_pkg.sv`. The systolic array dimensions (`PE_ROWS`, `PE_COLS`) control compute throughput.

**Multi-layer**: Instantiate N `transformer_decoder` blocks with a sequencer FSM that chains them. Weight memories can be shared (time-multiplexed) or replicated.

**FPGA targeting**: The design is synthesizable with minor modifications. Replace the combinational weight arrays with BRAM interfaces and the division operators in `layer_norm.sv` / `softmax_unit.sv` with shifts or reciprocal LUTs.

**Parallel softmax**: Instantiate N_HEADS `softmax_unit` modules in `multi_head_attention.sv` to process all heads simultaneously, reducing softmax latency from ~2,568 to ~642 cycles.

**Tiled projections**: Route the QKV and output projections through the 4×4 systolic array instead of the current combinational for-loops, reducing critical path and multiplier count.

## Implementation Estimates (Xilinx Artix-7)

| Block | DSP48 | FFs | LUTs | BRAM |
|-------|-------|-----|------|------|
| Processing Element | 1 | ~30 | ~20 | — |
| 4×4 Systolic Array | 16 | ~500 | ~350 | — |
| Softmax Unit (VEC_LEN=128) | 0 | ~2K | ~3K | — |
| Layer Normalisation | 2–4 | ~300 | ~500 | — |
| Multi-Head Attention | ~64 | ~5K | ~8K | ~32 KB |
| Feed-Forward Network | ~64 | ~3K | ~5K | — |
| **Full Decoder Block** | **~150** | **~10K** | **~15K** | **~32 KB** |

Clock frequency estimate: 100–200 MHz depending on place-and-route effort.

## License

MIT License. See `LICENSE` for details.
