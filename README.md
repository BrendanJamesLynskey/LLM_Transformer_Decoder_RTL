# LLM Transformer Decoder RTL Accelerator

A synthesizable **SystemVerilog** implementation of a Transformer Decoder block, optimized for LLM inference. Includes a full verification suite — behavioural Python golden model, SystemVerilog testbenches, and CocoTB testbenches — with 83 passing tests and no external Python dependencies for behavioural verification.

---

## Architecture

```
token_emb ──► [LayerNorm 1] ──► [Multi-Head Attention] ──►(+)──► [LayerNorm 2] ──► [Feed-Forward] ──►(+)──► out_emb
                                                            ▲  Residual 1                                ▲  Residual 2
                  └───────────────────────────────────────────┘              └───────────────────────────┘
```

This is a **pre-norm decoder** (GPT-2/LLaMA style): LayerNorm is applied before each sub-layer, residual connections bypass each sub-layer, and inference is autoregressive with KV-cache support.

---

## Key Parameters

| Parameter    | Default | Description                        |
|--------------|---------|------------------------------------|
| `D_MODEL`    | 64      | Embedding / model dimension        |
| `N_HEADS`    | 4       | Number of attention heads          |
| `D_HEAD`     | 16      | Per-head dimension (`D_MODEL / N_HEADS`) |
| `D_FF`       | 256     | FFN inner dimension (4 × D_MODEL)  |
| `MAX_SEQ_LEN`| 128     | Maximum sequence / context length  |
| `DATA_WIDTH` | 16      | Fixed-point word width (Q8.8)      |

All parameters are centralised in `rtl/transformer_pkg.sv`. Scaling up requires only changing values there.

---

## Fixed-Point Arithmetic

All computation uses **Q8.8 signed fixed-point** (16-bit) with 32-bit accumulation for MAC operations.

| Property    | Value                                        |
|-------------|----------------------------------------------|
| Format      | Q8.8 signed, 16-bit                          |
| Range       | −128.0 to +127.996                           |
| Resolution  | 1/256 ≈ 0.0039                               |
| Accumulator | 32-bit (full-precision MAC, truncated on output) |
| Softmax     | Piecewise-linear exponential approximation   |
| LayerNorm   | 32-entry RSqrt LUT with Newton–Raphson refinement; division replaced by right-shift |

Eliminating floating-point units drastically reduces area and power. 16×16-bit multipliers map directly to FPGA DSP48 primitives.

---

## Module Hierarchy

```
transformer_decoder          ← Top-level decoder block
├── layer_norm               ← Pre-attention & pre-FFN normalisation
├── multi_head_attention     ← Causal multi-head self-attention with KV-cache
├── feed_forward             ← Two-layer FFN with ReLU activation
├── softmax_unit             ← PWL-approximate softmax
├── systolic_array           ← Matrix-multiply engine
│   └── processing_element   ← Single MAC unit (systolic PE)
└── transformer_pkg          ← Parameters, types, FP utility functions
```

---

## Project Structure

```
├── rtl/
│   ├── transformer_pkg.sv        # Package: parameters, types, FP functions
│   ├── processing_element.sv     # Systolic PE (MAC unit)
│   ├── systolic_array.sv         # N×N systolic matrix multiply
│   ├── softmax_unit.sv           # PWL softmax approximation
│   ├── layer_norm.sv             # Layer normalisation
│   ├── multi_head_attention.sv   # Multi-head causal attention + KV-cache
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
│   ├── verify_behavioral.py      # Bit-accurate behavioural verification (54 tests)
│   └── lint_check.py             # RTL structural lint checker
├── docs/
│   ├── report.md                 # Technical report (Markdown)
│   └── report.pdf                # Technical report (PDF)
└── README.md
```

---

## Verification

The project has **83 passing tests** across three verification tiers.

### Tier 1 — Behavioural Verification (no simulator required)

A bit-accurate Python model mirrors the RTL using identical Q8.8 arithmetic. Requires only **Python 3.8+** with no additional dependencies.

```bash
python3 scripts/verify_behavioral.py
```

Runs **54 behavioural tests** with golden-model comparison:

| Module                  | Tests | Key Checks                                        |
|-------------------------|-------|---------------------------------------------------|
| Fixed-Point Utilities   | 16    | Roundtrip, multiply, saturating add               |
| Processing Element      | 9     | MAC, forwarding, clear, 20-op random golden model |
| Systolic Array          | 8     | Single element, 2×2 matmul `[[19,22],[43,50]]`, clear |
| Softmax Unit            | 8     | Uniform, dominant, ordering, sum≈1.0, back-to-back |
| Layer Normalisation     | 5     | Constant→zero, symmetry, γ/β scaling, centering  |
| Feed-Forward Network    | 4     | ReLU zeroing, bias propagation                    |
| Decoder Integration     | 4     | Full pipeline, KV-cache, 2-token sequential       |

### Tier 2 — RTL Simulation (requires iverilog)

**29 RTL simulation tests** via SystemVerilog testbenches:

```bash
# Run all RTL tests
./scripts/run_sim.sh all

# Run individual module tests
./scripts/run_sim.sh pe         # Processing Element
./scripts/run_sim.sh systolic   # Systolic Array
./scripts/run_sim.sh softmax    # Softmax Unit
./scripts/run_sim.sh decoder    # Full Decoder (integration)
```

### Tier 3 — CocoTB (Python-driven RTL testbenches)

```bash
cd tb/cocotb

make -f Makefile.pe       # Processing Element
make -f Makefile.softmax  # Softmax Unit
```

### Structural Lint

```bash
python3 scripts/lint_check.py
```

Checks module/endmodule balance, package imports, reset patterns, and cross-file instantiation resolution across all 8 RTL files.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.8 | Behavioural verification (no extra packages needed) |
| Icarus Verilog | ≥ 12.0 | RTL simulation (SystemVerilog 2012) |
| CocoTB | ≥ 1.8 | Python-driven RTL testbenches |

```bash
# Ubuntu / Debian
sudo apt-get install iverilog
pip install cocotb

# macOS
brew install icarus-verilog
pip install cocotb
```

---

## Design Rationale

### Q8.8 Fixed-Point

Integer-only datapaths eliminate floating-point units, substantially reducing area and power. Q8.8 provides sufficient dynamic range for small-model inference while keeping multipliers at 16×16 bits — a natural fit for FPGA DSP48 blocks.

### Systolic Array

The systolic architecture maximises data reuse: each operand traverses N PEs, yielding O(N²) compute with O(N) I/O bandwidth. This maps directly to the matrix multiplications that dominate transformer inference.

### Pre-Norm

LayerNorm before each sub-layer (rather than after) improves training stability and matches the architecture used by GPT-2, LLaMA, and most modern decoder-only LLMs. At inference, results are equivalent.

### KV-Cache

During autoregressive generation only the current token's **Q** is computed fresh; **K** and **V** from all prior positions are cached. This reduces per-token attention compute from O(n²·d) to O(n·d).

### LayerNorm Implementation

Standard LayerNorm requires a reciprocal square root, which is expensive in integer logic. This design replaces the division with a right-shift and uses a 32-entry RSqrt LUT with a single Newton–Raphson refinement step, providing sufficient accuracy for Q8.8 precision without a hardware divider.

---

## Architectural Variants

Two area/throughput trade-off configurations are documented in `docs/report.md`:

| Variant | Description | Register count |
|---------|-------------|----------------|
| High-throughput | Register-bridge architecture; maximises pipelining | Baseline |
| Minimum-area | Streaming architecture; weight/activation reuse | −99.4% registers |

---

## Extending the Design

**Scaling**: Increase `D_MODEL`, `N_HEADS`, `D_FF` in `transformer_pkg.sv`. The systolic array dimensions (`PE_ROWS`, `PE_COLS`) control throughput.

**Multi-layer**: Instantiate N `transformer_decoder` blocks with a sequencer FSM. Weight memories can be shared (time-multiplexed) or replicated per layer.

**FPGA targeting**: The design is synthesizable as-is for Xilinx/Intel FPGAs. Replace combinational weight arrays with BRAM interfaces for practical implementations beyond the reference parameter set.

**Quantisation upgrade**: The package-level type definitions and fixed-point utility functions in `transformer_pkg.sv` are the single point of change for moving to wider formats (e.g. Q12.4, INT8 with per-channel scale).

---

## License

MIT License. See `LICENSE` for details.

---

## Synthesis Results

Target: Xilinx Artix-7 (xc7a35tcpg236-1) | Tool: Vivado 2025.2

The full top-level was decomposed into sub-modules for synthesis due to host memory constraints (16 GB). Sub-module figures are not additive — shared logic and optimisations at the top level mean the true total would differ.

| Module | LUTs | FFs | BRAM | DSP | Fmax (MHz) |
|--------|------|-----|------|-----|------------|
| **processing_element** | 34 | 64 | 0 | 1 | 229.9 |
| **systolic_array** (4×4) | 455 | 901 | 0 | 16 | 156.8 |
| **softmax_unit** | 537 | 584 | 0 | 3 | 47.9 |
| **layer_norm** | 1,585 | 2,157 | 0 | 7 | 29.2 |
| transformer_decoder_top (full) | — | — | — | — | Exceeded 10 GB memory limit |

*Auto-generated by Vivado batch synthesis. Clock target: 100 MHz. Major sub-modules shown in **bold**. Default parameterisation: D_MODEL=64, N_HEADS=4, PE_ROWS=PE_COLS=4, Q8.8 fixed-point.*
