# LLM Transformer Decoder RTL Accelerator

A synthesizable SystemVerilog implementation of a **Transformer Decoder block** optimized for LLM inference, featuring BRAM-backed weight storage, reciprocal-LUT softmax, KV-cache, and comprehensive verification.

## Architecture Overview

```
                    ┌──────────────────────────────────────────┐
  wl_en/addr/data ──►  transformer_decoder_top                 │
                    │                                          │
                    │  ┌─────────────────────────────────────┐ │
  INIT_FILE ──────────►│  Weight BRAMs (12 instances)         │ │
                    │  │  Wq Wk Wv Wo W1 W2 b1 b2 LN γ/β   │ │
                    │  └──────────────┬──────────────────────┘ │
                    │                 │ register arrays         │
                    │  ┌──────────────▼──────────────────────┐ │
  token_emb ────────►  │  transformer_decoder                 │ │
                    │  │                                      │ │
                    │  │  ──► [LN1] ──► [Multi-Head Attn] ◄──┤ │
                    │  │  │              │ ┌────────────┐     │ │
                    │  │  │              ├─┤ softmax    │     │ │
                    │  │  │              │ │ (LUT+NR)   │     │ │
                    │  │  │              │ └────────────┘     │ │
                    │  │  └──── (+) ◄────┘                    │ │
                    │  │         │                             │ │
                    │  │  [LN2] ──► [FFN] ──► (+) ──► out_emb│ │
                    │  └──────────────────────────────────────┘ │
                    │                                          │
                    │  ┌──────────────────────────────────────┐ │
                    │  │  KV-Cache BRAMs (2× dual-port)       │ │
                    │  │  K: 128×64  V: 128×64                │ │
                    │  └──────────────────────────────────────┘ │
                    └──────────────────────────────────────────┘
```

This is a **pre-norm** decoder architecture (GPT-2/LLaMA style) implementing autoregressive inference with BRAM-backed weights, KV-cache, and division-free softmax via reciprocal LUT.

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
transformer_decoder_top          (BRAM-backed synthesis top-level)
├── bram_sp ×12                  (Weight/bias/parameter storage)
├── kv_cache_bram ×2             (K and V caches)
│   └── bram_dp                  (True dual-port BRAM)
└── transformer_decoder          (Decoder compute core)
    ├── layer_norm ×2            (Pre-attention & pre-FFN normalisation)
    ├── multi_head_attention     (Causal multi-head self-attention)
    │   └── softmax_unit         (Reciprocal-LUT softmax, time-multiplexed)
    ├── feed_forward             (Two-layer FFN with ReLU)
    ├── systolic_array           (Matrix multiply engine)
    │   └── processing_element   (Single MAC unit)
    └── transformer_pkg          (Parameters, types, FP utilities)

Utility modules (not instantiated in top, available for future use):
    weight_bram                  (2D weight matrix w/ column-read FSM)
```

## Weight Initialisation and Loading

Weights can be loaded in two ways:

**At synthesis/simulation start** — pass hex file paths to the `INIT_FILE` parameters of `transformer_decoder_top`:

```systemverilog
transformer_decoder_top #(
  .WQ_INIT("weights/wq.hex"),
  .WK_INIT("weights/wk.hex"),
  // ... etc
) u_top ( ... );
```

Hex files contain one 16-bit value per line in row-major order.

**At runtime** — use the weight-loading bus to write individual elements:

```
wl_en   = 1;
wl_addr = 16'h0042;  // Wq[1][2] (row 1, col 2 = 1*64+2 = 66 = 0x42)
wl_data = 16'h0100;  // 1.0 in Q8.8
```

The unified address space maps all 49,344 weight words:

| Region | Base | Size | Dimensions |
|--------|------|------|------------|
| Wq | 0x0000 | 4096 | 64 × 64 |
| Wk | 0x1000 | 4096 | 64 × 64 |
| Wv | 0x2000 | 4096 | 64 × 64 |
| Wo | 0x3000 | 4096 | 64 × 64 |
| FFN W1 | 0x4000 | 16384 | 64 × 256 |
| FFN W2 | 0x8000 | 16384 | 256 × 64 |
| LN1 γ | 0xC000 | 64 | 64 |
| LN1 β | 0xC040 | 64 | 64 |
| LN2 γ | 0xC080 | 64 | 64 |
| LN2 β | 0xC0C0 | 64 | 64 |
| FFN b1 | 0xC100 | 256 | 256 |
| FFN b2 | 0xC200 | 64 | 64 |

## BRAM Architecture

### bram_sp — Generic Single-Port BRAM

Parameterised by `DATA_WIDTH`, `DEPTH`, and `INIT_FILE`. Synchronous read-first behavior with 1-cycle read latency. Synthesises to Xilinx BRAM36 / Intel M20K primitives.

### bram_dp — Generic True Dual-Port BRAM

Two fully independent read/write ports. Used for KV-cache where port A writes new K/V vectors and port B reads during attention scoring.

### kv_cache_bram — KV-Cache Controller

Wraps `bram_dp` for a MAX_SEQ_LEN × D_MODEL cache. Provides a vector-write interface (auto-incrementing FSM writes all D_MODEL elements) and element-level random reads at `(position, dimension)`.

### weight_bram — 2D Weight Matrix with Column-Read FSM

Wraps `bram_sp` to store a ROWS × COLS matrix. Provides a column-read interface matching the compute modules' access pattern: request column index, receive the full column in a registered buffer after ROWS+1 cycles.

## Attention Pipeline

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

| Stage | Operation | Cycles (seq_pos=127) |
|-------|-----------|---------------------|
| `S_PROJ_QKV` | Project input → Q, K, V | 64 |
| `S_WRITE_CACHE` | Store K, V in KV-cache | 1 |
| `S_SCORE` | Compute Q·K^T / √d_k | 512 |
| `S_SOFTMAX_PREP` | Pad scores with −8.0 (causal mask) | 1 per head |
| `S_SOFTMAX_RUN` | Softmax: find-max → exp → sum → reciprocal → multiply | ~641 per head |
| `S_SOFTMAX_STORE` | Copy probabilities | 1 per head |
| `S_WEIGHTED_SUM` | Probability-weighted sum of V | 4 |
| `S_OUTPUT_PROJ` | Project through Wo | 64 |
| **Total** | | **~3,214** |

### Softmax (Division-Free)

The softmax normalisation step `probs[i] = exp[i] / Σexp` is implemented without a hardware divider:

1. **32-entry reciprocal LUT** (Q2.14) indexed by the top 5 mantissa bits of the sum after CLZ normalisation to [0.5, 1.0)
2. **One Newton-Raphson iteration**: `r₁ = r₀ × (2 − sum_norm × r₀)`, doubling precision to ~12 bits
3. **Denormalising shift**: `recip = r₁ >> (14 − lz)` to produce `65536 / exp_sum`
4. **Multiply**: `probs[i] = fp_mul(exp[i], recip)` using the standard Q8.8 multiplier

This replaces the non-synthesizable `/` operator with a 512-bit ROM, two 16×16 multipliers, and a barrel shifter. Max error vs exact division: ±1 LSB (0.0039).

## Fixed-Point Arithmetic

All computation uses **Q8.8 signed fixed-point** (16-bit) with 32-bit accumulators:

| Property | Value |
|----------|-------|
| Format | 1 sign + 7 integer + 8 fractional bits |
| Range | −128.0 to +127.996 |
| Resolution | 1/256 ≈ 0.0039 (~48 dB SQNR) |
| Multiply | 16×16 → 32-bit product, arithmetic right-shift by 8 |
| Accumulate | Full 32-bit precision, truncated on output |
| Addition | Saturating (clamps to representable range) |
| Softmax exp | 4-segment piecewise-linear over [−8, 0] |
| Softmax 1/Σ | 32-entry reciprocal LUT + Newton-Raphson |
| LayerNorm 1/√x | 4-entry LUT in transformer_pkg |

## Project Structure

```
├── rtl/
│   ├── transformer_pkg.sv          # Parameters, types, FP functions
│   ├── processing_element.sv       # Systolic PE (MAC unit)
│   ├── systolic_array.sv           # NxN systolic matrix multiply
│   ├── softmax_unit.sv             # Reciprocal-LUT softmax
│   ├── layer_norm.sv               # Layer normalisation
│   ├── multi_head_attention.sv     # Multi-head attention + softmax + KV cache
│   ├── feed_forward.sv             # Position-wise FFN (ReLU)
│   ├── transformer_decoder.sv      # Decoder compute core
│   ├── bram_sp.sv                  # Generic single-port BRAM
│   ├── bram_dp.sv                  # Generic true dual-port BRAM
│   ├── weight_bram.sv              # 2D weight matrix w/ column-read FSM
│   ├── kv_cache_bram.sv            # KV-cache controller (dual-port)
│   └── transformer_decoder_top.sv  # BRAM-backed synthesis top-level
├── tb/
│   ├── sv/
│   │   ├── tb_processing_element.sv
│   │   ├── tb_systolic_array.sv
│   │   ├── tb_softmax.sv
│   │   ├── tb_bram.sv
│   │   └── tb_transformer_decoder.sv
│   └── cocotb/
│       ├── test_processing_element.py
│       ├── test_softmax.py
│       ├── Makefile.pe
│       └── Makefile.softmax
├── scripts/
│   ├── run_sim.sh                  # Simulation runner (iverilog)
│   ├── verify_behavioral.py        # Bit-accurate behavioural verification (54 tests)
│   └── lint_check.py               # RTL structural lint checker
├── docs/
│   ├── report.md
│   └── report.pdf
└── README.md
```

## Verification

The project has three verification layers: a bit-accurate Python model (54 tests), structural RTL lint (13 files), and iverilog RTL simulation (5 testbenches).

### Behavioural Model (Python — no simulator needed)

```bash
python3 scripts/verify_behavioral.py
```

**54 tests**, all passing, across every module:

| Module | Tests | Key Checks |
|--------|-------|------------|
| Fixed-Point Utilities | 16 | Roundtrip, multiply, saturating add, edge values |
| Processing Element | 9 | MAC, forwarding, clear, 20-op randomised golden model |
| Systolic Array | 8 | Single element, 2×2 matmul `[[19,22],[43,50]]`, clear |
| Softmax Unit | 8 | Uniform, dominant, ordering, sum≈1.0, back-to-back (reciprocal LUT) |
| Layer Normalisation | 5 | Constant→zero, symmetry, gamma/beta, centering |
| Feed-Forward Network | 4 | ReLU zeroing, bias propagation |
| Decoder Integration | 12 | Full pipeline w/ softmax, causal mask, KV-cache, 2-token sequential |

The softmax tests validate the reciprocal-LUT normalisation matches exact division to within ±1 LSB. The decoder integration tests verify single-position probability ≈ 0.90, multi-position sum ≈ 0.95, and future-position masking < 0.01.

### RTL Lint

```bash
python3 scripts/lint_check.py
```

Validates all 13 RTL files: module balance, package imports, reset patterns, instantiation resolution. Reports 0 errors, 5 warnings (BRAM blocks intentionally lack reset in their memory `always_ff` — standard practice for inference to FPGA BRAM primitives).

### RTL Simulation (iverilog)

#### Prerequisites

- **Python** ≥ 3.8
- **Icarus Verilog** ≥ 12.0

```bash
sudo apt-get install iverilog   # Ubuntu/Debian
brew install icarus-verilog      # macOS
```

#### Test Results

| Testbench | Compile | Result |
|-----------|---------|--------|
| tb_processing_element | ✓ | **6/6 PASS** |
| tb_systolic_array | ✓ | **4/4 PASS** |
| tb_softmax | ✓ | **4/4 PASS** |
| tb_bram | ✓ | **8/8 PASS** |
| tb_transformer_decoder | ✓ | **4/4 PASS** |
| Full hierarchy (13 files) | ✓ | **Compiles clean** |

**Total: 80/80 tests pass** (54 behavioural + 26 RTL simulation).

All module ports use packed arrays (`logic signed [N-1:0][W-1:0]`) for 1D vector signals, ensuring correct value propagation in iverilog. Multi-dimensional weight matrices remain as unpacked arrays (set before simulation start).

#### Running Tests

```bash
# Individual testbenches
iverilog -g2012 -o sim_pe rtl/transformer_pkg.sv rtl/processing_element.sv tb/sv/tb_processing_element.sv && vvp sim_pe

iverilog -g2012 -o sim_bram rtl/transformer_pkg.sv rtl/bram_sp.sv rtl/bram_dp.sv tb/sv/tb_bram.sv && vvp sim_bram

# Full decoder integration
iverilog -g2012 -o sim_decoder \
  rtl/transformer_pkg.sv rtl/bram_sp.sv rtl/bram_dp.sv \
  rtl/processing_element.sv rtl/systolic_array.sv \
  rtl/softmax_unit.sv rtl/layer_norm.sv \
  rtl/multi_head_attention.sv rtl/feed_forward.sv \
  rtl/transformer_decoder.sv \
  tb/sv/tb_transformer_decoder.sv && vvp sim_decoder

# Full hierarchy (compilation check)
iverilog -g2012 -o sim_top \
  rtl/transformer_pkg.sv rtl/bram_sp.sv rtl/bram_dp.sv \
  rtl/processing_element.sv rtl/systolic_array.sv \
  rtl/softmax_unit.sv rtl/layer_norm.sv \
  rtl/multi_head_attention.sv rtl/feed_forward.sv \
  rtl/transformer_decoder.sv rtl/kv_cache_bram.sv \
  rtl/weight_bram.sv rtl/transformer_decoder_top.sv
```

### CocoTB Tests

```bash
cd tb/cocotb
make -f Makefile.pe        # Processing Element
make -f Makefile.softmax   # Softmax
```

## Design Decisions

### Why Q8.8 Fixed-Point?
Integer-only datapaths eliminate floating-point units, reducing area and power. Q8.8 keeps multipliers at 16×16 bits — ideal for FPGA DSP48 blocks.

### Why Reciprocal LUT Instead of Division?
Hardware dividers are either non-synthesizable (Verilog `/`) or require multi-cycle iterative circuits. The 32-entry LUT + one Newton-Raphson iteration computes the reciprocal in a single cycle using only a small ROM, two multipliers, and a shifter. Accuracy is ±1 LSB, matching the Q8.8 output precision.

### Why BRAM for Weights?
Combinational array ports (the original design) are fine for simulation but synthesise to massive LUT-based register files. BRAM storage reduces this from ~800K flip-flops to ~100 KB of block RAM (a fraction of even a small FPGA). The `INIT_FILE` parameter enables preloading from hex files at synthesis.

### Why Time-Multiplexed Softmax?
A single softmax_unit shared across 4 heads saves ~3× area at the cost of 4× latency. Parallelising to N_HEADS instances is a one-line change when latency is the bottleneck.

### Why Pre-Norm?
Pre-norm (LayerNorm before attention/FFN) matches GPT-2, LLaMA, and most modern LLMs. It is more stable and produces identical results at inference time.

### KV-Cache Strategy
Only the current token's Q is computed fresh; K and V from all prior positions are cached in dual-port BRAM, reducing per-token attention compute from O(n²·d) to O(n·d).

## Use Cases and Applicability

### Where This Design Fits

This accelerator implements a **single decoder layer for autoregressive inference** — the core primitive of GPT-style text generation. Understanding where it applies (and where it doesn't) helps frame its value.

### Inference — Primary Target

This design is purpose-built for **LLM inference on edge and embedded FPGA platforms**. Several architectural choices reflect this:

**Weight-stationary dataflow.** Weights are loaded once into BRAM and reused across every token. There is no gradient storage, no optimizer state, no backward pass — the entire memory budget goes to weights and KV-cache. This is the fundamental difference between inference and training hardware.

**KV-cache with incremental updates.** Each new token writes one K/V vector and reads all prior positions. The dual-port BRAM design (port A writes, port B reads) supports this single-token-at-a-time pattern directly. Production inference engines like vLLM and TensorRT-LLM use exactly this pattern, managing cache memory across requests.

**Fixed-point arithmetic.** Q8.8 quantisation reduces multiplier area to single DSP48 slices. Post-training quantisation to INT8 or INT4 is standard practice in production inference (GPTQ, AWQ, SmoothQuant). This design demonstrates the hardware-side implementation of quantised inference.

**Low-latency single-token generation.** The design processes one token per pipeline pass (~3,200 cycles at 128 sequence length), targeting interactive generation where latency per token matters more than throughput.

Concrete deployment scenarios include on-device language models for IoT/robotics (running small transformer models entirely on FPGA with no host CPU dependency), FPGA inference cards for data centre offload (Xilinx Alveo or Intel Stratix with multiple decoder layers), and custom ASIC prototyping (using the RTL as a starting point for a tape-out targeting inference workloads).

### Low-Power and Edge Deployment — Strong Fit

The design is well-suited to **power-constrained environments** for several reasons:

**No external memory bandwidth.** With weights in on-chip BRAM (~128 KB), inference requires zero DRAM accesses for a small model. DRAM access typically dominates power consumption in neural network inference — eliminating it can reduce power by 10–100×. For the current D_MODEL=64 configuration, the entire model fits on-chip.

**Clock gating opportunity.** The FSM-driven sequential architecture naturally enables clock gating: only the active module draws dynamic power in any given cycle. The systolic array, softmax, LayerNorm, and FFN are never active simultaneously.

**Deterministic latency.** Fixed pipeline depth with no data-dependent branching means power draw is predictable and consistent — important for battery-powered or energy-harvesting applications.

**Scaling consideration.** Production LLMs (7B+ parameters) far exceed on-chip BRAM capacity. For larger models, this architecture would require an external memory interface (HBM/DDR) with a weight-streaming controller, at which point power advantages diminish. The sweet spot is small specialised models (1M–50M parameters) that fit entirely on-chip.

### Training — Not Applicable

This design **does not support training** and is not easily adapted for it. The fundamental gaps are:

**No backward pass.** Training requires computing gradients through every layer via backpropagation. This needs either: (a) storing all intermediate activations for the backward pass (enormous memory), or (b) recomputation (doubling latency). Neither is implemented.

**No floating-point support.** Training is highly sensitive to numerical precision — modern training uses BF16 or FP32 accumulation. Q8.8 fixed-point lacks the dynamic range for gradient computation, where values can span many orders of magnitude. Straight-through estimators and loss scaling can partially compensate, but Q8.8 is too narrow for stable training.

**No weight update logic.** Training requires an optimiser (SGD, Adam) that reads gradients, maintains momentum/variance state, and writes updated weights. This is an entirely separate datapath.

**No batch support.** Training throughput depends on processing many samples simultaneously. This design processes one token at a time with no batching.

Training accelerators (GPU, TPU, Cerebras WSE) are architecturally very different: they prioritise memory bandwidth, floating-point throughput, and all-reduce communication over the low-latency single-inference path optimised here.

### Prefill vs Decode Phases

Modern LLM serving splits inference into two phases with different computational profiles:

**Prefill (prompt processing)** is compute-bound — it processes all prompt tokens in parallel through matrix multiplications. This design's sequential token-by-token architecture is suboptimal for prefill; a batched matrix engine would be more efficient.

**Decode (token generation)** is memory-bandwidth-bound — it generates one token at a time, reading the full KV-cache each step. This is exactly what the design optimises for: single-token processing with on-chip KV-cache reuse.

A production system would pair this decode engine with a separate prefill accelerator, or add a batch-mode to the systolic array for prefill.

### Summary

| Scenario | Fit | Notes |
|----------|-----|-------|
| Edge/FPGA inference (small models) | ★★★★★ | Primary design target. On-chip weights, low power, deterministic latency |
| ASIC inference prototyping | ★★★★☆ | Proven architecture; add memory interfaces for larger models |
| Low-power / battery-constrained | ★★★★☆ | Zero DRAM for small models; FSM enables clock gating |
| Data centre decode engine | ★★★☆☆ | Architecture sound; needs HBM interface and multi-layer stacking |
| Prefill / prompt processing | ★★☆☆☆ | Sequential design; would need batch-mode systolic operation |
| Training | ☆☆☆☆☆ | Fundamentally different requirements (FP, backward pass, optimiser) |
| Fine-tuning / LoRA | ★☆☆☆☆ | Could serve as frozen forward-pass engine with external adapter logic |

## Extending the Design

**Streaming weight access**: Refactor compute modules to use address/data interfaces instead of array ports, reading weights directly from BRAM one element per cycle. This eliminates the register arrays in `transformer_decoder_top`.

**Parallel softmax**: Instantiate N_HEADS `softmax_unit` modules for 4× lower softmax latency.

**Tiled projections**: Route QKV and output projections through the 4×4 systolic array for reduced critical path.

**Multi-layer stacking**: See dedicated section below.

**Scaling up**: Increase `D_MODEL`, `N_HEADS`, `D_FF` in `transformer_pkg.sv`.

## Multi-Layer Stacking

A production LLM uses many identical decoder layers (GPT-2: 12–48, LLaMA-7B: 32, LLaMA-70B: 80). This design implements a single layer. Stacking multiple layers to form a complete model can follow several strategies, each with different area/performance/memory trade-offs.

### Strategy 1: Spatial Replication (One Layer Per Instance)

The most straightforward approach: instantiate N separate `transformer_decoder_top` modules, each with its own weight BRAMs, and chain their outputs to inputs.

```
token_emb ──► [Layer 0] ──► [Layer 1] ──► ... ──► [Layer N-1] ──► logits
               128 KB         128 KB                  128 KB
```

**Wiring.** Each layer's `out_emb` connects directly to the next layer's `token_emb`. A top-level sequencer asserts `start` on layer 0, waits for `valid`, then asserts `start` on layer 1, and so on. The `seq_pos` signal is shared across all layers (same token position in every layer).

**KV-cache.** Each layer maintains its own independent KV-cache BRAM pair. This is correct — in a transformer, each layer's attention operates on its own K/V projections, so caches are not shared between layers.

**Resource cost.** Linear in N: an N-layer model requires N × 126 BRAM18K (~128 KB each). A 12-layer GPT-2-small-scale model at D_MODEL=64 would need ~1,512 BRAM18K and ~1.5 MB — feasible on a Xilinx Kintex UltraScale (1,800+ BRAM18K) or Alveo U250 (5,376 BRAM18K). Compute resources (DSP48, LUTs) also scale linearly.

**Latency.** N × single-layer latency (~3,200 cycles per layer at seq_len=128). All layers execute sequentially since each depends on the previous layer's output. For 12 layers: ~38,400 cycles = ~384 μs at 100 MHz.

**When to use.** When the FPGA is large enough to hold all layers simultaneously. This gives the simplest control logic and the lowest latency since there is no weight reloading overhead.

### Strategy 2: Temporal Reuse (Single Instance, Weight Swapping)

Use a single `transformer_decoder_top` instance and reload its weight BRAMs between layers.

```
                    ┌──────────────────────────┐
token_emb ──►       │  transformer_decoder_top  │ ──► out_emb
                    │  (single instance)        │       │
                    └──────────────────────────┘       │
                         ▲                              │
                         │ weight-load bus              │
              ┌──────────┴──────────┐          ┌───────▼───────┐
              │  External Memory    │          │  Embedding    │
              │  (DDR/HBM/Flash)   │          │  Register     │
              │  Layer 0 weights   │          │  (feedback)   │
              │  Layer 1 weights   │          └───────────────┘
              │  ...               │
              └─────────────────────┘
```

**Weight loading.** Between each layer pass, a DMA controller streams the next layer's 49,344 weights through the `wl_en`/`wl_addr`/`wl_data` bus. At one word per cycle, this takes ~49,344 cycles. With a wider bus (32-bit or 64-bit data), this halves or quarters.

**Embedding feedback.** The `out_emb` output is registered and fed back to `token_emb` for the next layer pass. A simple output register with a mux (external input for layer 0, feedback for layers 1+) handles this.

**KV-cache management.** This is the main complication. Each layer needs its own KV-cache, but the single instance only has one pair of cache BRAMs. Options: (a) save/restore cache contents to external memory between layers (expensive — 16 KB per layer per swap), (b) use external memory for all KV-cache storage with an address offset per layer, or (c) keep N separate KV-cache BRAM pairs while sharing the compute core (hybrid approach, see Strategy 3).

**Resource cost.** Constant compute resources (1× decoder = ~126 BRAM18K for weights). KV-cache storage depends on the approach chosen. External memory bandwidth becomes the bottleneck.

**Latency.** N × (compute_time + weight_load_time). For 12 layers with 49K-cycle reload: ~12 × (3,200 + 49,344) = ~630,528 cycles. The weight reload dominates — the design spends 94% of its time loading weights. A wider weight bus or block-RAM DMA is essential to make this practical.

**When to use.** When the target FPGA cannot fit multiple layer instances. Requires external memory with sufficient bandwidth. Best paired with the streaming weight architecture (see Extending the Design) which eliminates the register-array pre-load.

### Strategy 3: Hybrid (Shared Compute, Dedicated KV-Cache)

A middle ground: one compute core with weight swapping, but N dedicated KV-cache BRAM pairs that persist across all tokens.

```
                    ┌─────────────────────────────────────┐
                    │  multi_layer_top                     │
                    │                                      │
                    │  ┌────────────────────────────────┐ │
 token_emb ────────►│  │ transformer_decoder (shared)    │ │
                    │  │ + Weight BRAMs (swapped/layer)  │ │
                    │  └───────────┬────────────────────┘ │
                    │              │                       │
                    │  ┌───────────▼────────────────────┐ │
                    │  │ KV-Cache Bank                   │ │
                    │  │ Layer 0: K₀[128×64] V₀[128×64] │ │
                    │  │ Layer 1: K₁[128×64] V₁[128×64] │ │
                    │  │ ...                              │ │
                    │  │ Layer N: Kₙ[128×64] Vₙ[128×64] │ │
                    │  └──────────────────────────────────┘ │
                    └─────────────────────────────────────┘
```

**How it works.** The compute core processes layer 0, writing to KV-cache bank 0 and reading from KV-cache bank 0. Then weights are swapped and it processes layer 1 with KV-cache bank 1, and so on. A layer-index register selects which cache bank connects to the decoder's cache ports.

**KV-cache cost.** N layers × 2 BRAMs × ~8 BRAM18K = 16N BRAM18K. For 12 layers: 192 BRAM18K just for cache. Combined with the shared compute BRAMs (~126): ~318 BRAM18K total.

**Advantage over Strategy 2.** No cache save/restore traffic. The KV-cache is the state that grows with sequence length and must persist across every token generation step — keeping it on-chip avoids the most latency-sensitive external memory accesses.

**When to use.** When the FPGA has enough BRAM for N cache banks but not N full layer instances. This is often the right trade-off: cache BRAMs (16N BRAM18K) are much cheaper than full weight BRAMs (126N BRAM18K).

### Strategy 4: Pipelined Layers

If the FPGA can fit N layer instances, overlap execution across tokens rather than running them sequentially.

```
Token t:    [L0]──►[L1]──►[L2]──►...──►[LN]──► output
Token t+1:         [L0]──►[L1]──►[L2]──►...──►[LN]──► output
Token t+2:                [L0]──►[L1]──►...
```

Once layer 0 finishes token t and passes its output to layer 1, layer 0 can immediately begin processing token t+1. This creates a pipeline with throughput of one token per single-layer latency (~3,200 cycles) rather than one token per N-layer latency.

**Requirement.** Each layer needs independent weight BRAMs (same as Strategy 1) AND each layer's KV-cache must support concurrent read (for the current token) and write (for the new token). The dual-port BRAM already supports this.

**Complication.** Autoregressive generation has a data dependency: the next token depends on the final layer's output (via the language model head and sampling). So the pipeline only helps with throughput if processing a batch of independent sequences, or during prefill when all tokens are known in advance. For single-sequence autoregressive decode, pipelining provides no benefit — each token must complete all N layers before the next token's identity is known.

**When to use.** Batch inference or prefill, where multiple independent tokens can be in-flight simultaneously. Not useful for single-sequence autoregressive generation.

### Comparison

| Strategy | BRAM18K (12 layers) | Latency/token | Weight bandwidth | Complexity |
|----------|---------------------|---------------|-----------------|------------|
| Spatial replication | ~1,512 | ~38K cycles | None (all on-chip) | Low |
| Temporal reuse | ~126 + ext. mem | ~630K cycles | ~600K words/token | Medium |
| Hybrid (shared compute) | ~318 + ext. mem | ~630K cycles | ~600K words/token | Medium |
| Pipelined | ~1,512 | ~3.2K cycles* | None (all on-chip) | High |

\* Throughput per token in steady state; single-sequence autoregressive decode still takes ~38K cycles.

### Practical Implementation Notes

**Top-level sequencer.** All strategies need a state machine that manages the layer progression: tracking the current layer index, asserting start/waiting for valid on each layer pass, managing the embedding feedback path, and (for weight-swapping strategies) triggering weight reloads between layers.

**Embedding register.** A D_MODEL-wide register between layers stores the intermediate embedding. For Strategy 1, this is just wiring; for Strategies 2–3, it requires an explicit register with feedback mux.

**Layer-norm final.** Production transformers apply a final LayerNorm after the last decoder layer, before the language model head. This would be one additional `layer_norm` instance at the output of the layer stack.

**Language model head.** After all N decoder layers, the output embedding is projected to vocabulary logits via a large matrix multiply (D_MODEL × VOCAB_SIZE). This is a single linear layer with no bias, often sharing weights with the input embedding table. For D_MODEL=64, VOCAB_SIZE=256: 16K parameters, fitting in one additional BRAM.

## Implementation Estimates (Xilinx Artix-7)

| Block | DSP48 | FFs | LUTs | BRAM18K |
|-------|-------|-----|------|---------|
| Processing Element | 1 | ~30 | ~20 | — |
| 4×4 Systolic Array | 16 | ~500 | ~350 | — |
| Softmax Unit (VEC_LEN=128) | 2 | ~2K | ~3K | — |
| Layer Normalisation | 2–4 | ~300 | ~500 | — |
| Multi-Head Attention | ~64 | ~5K | ~8K | — |
| Feed-Forward Network | ~64 | ~3K | ~5K | — |
| Weight BRAMs (12 instances) | — | — | — | ~110 |
| KV-Cache BRAMs (2 instances) | — | — | — | ~16 |
| **Full Decoder (top)** | **~150** | **~11K** | **~17K** | **~126** |

Clock frequency estimate: 100–200 MHz depending on place-and-route effort.

## License

MIT License. See `LICENSE` for details.
