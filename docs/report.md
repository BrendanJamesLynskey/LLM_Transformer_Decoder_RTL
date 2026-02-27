# LLM Transformer Decoder RTL Accelerator — Technical Report

## 1. Introduction

This report documents the design, implementation, and verification of a synthesizable RTL accelerator for LLM (Large Language Model) transformer decoder inference. The accelerator implements a single transformer decoder block — the fundamental repeating unit of GPT-style language models — in SystemVerilog, targeting FPGA and ASIC deployment.

The design prioritizes inference efficiency through fixed-point arithmetic, systolic array compute, BRAM-backed weight and cache storage, division-free softmax normalisation, and KV-cache reuse, reflecting the architectural patterns used in production LLM inference engines.


## 2. Background

### 2.1 Transformer Decoder Architecture

The transformer decoder processes tokens autoregressively: given a sequence of prior tokens, it predicts the next token. Each decoder layer applies two sub-layers with residual connections. First, Multi-Head Self-Attention computes attention scores between the current query and all prior key/value pairs, enabling the model to attend to relevant context. Second, a Feed-Forward Network (FFN) applies a two-layer MLP with non-linear activation independently to each position. Both sub-layers are preceded by Layer Normalization (pre-norm architecture) and followed by residual additions.

### 2.2 Inference vs Training

This accelerator targets inference only, which has distinct characteristics compared to training. Batch sizes are typically 1 for interactive generation. Only the forward pass is needed with no backpropagation. KV-cache eliminates redundant attention computation. Weights are fixed, enabling weight-stationary dataflows and BRAM preloading.

### 2.3 Fixed-Point Quantization

Modern LLM inference widely uses quantization to reduce memory bandwidth and compute cost. Our Q8.8 fixed-point format (8 integer bits, 8 fractional bits) provides a range of -128.0 to +127.996 with 1/256 resolution, which is sufficient for demonstrating the architectural concepts while keeping multiplier width at 16 bits.


## 3. Architecture

### 3.1 Top-Level Block Diagram

The design has two levels of hierarchy. The synthesis top-level (`transformer_decoder_top`) wraps the compute core with BRAM-backed storage for all 49,344 weight parameters and 16,384 KV-cache entries. The compute core (`transformer_decoder`) implements the pre-norm architecture used by GPT-2, LLaMA, and most modern LLMs: LayerNorm 1, Multi-Head Attention with KV-cache, residual addition, LayerNorm 2, Feed-Forward Network with ReLU, and a second residual addition.

### 3.2 Module Hierarchy

The full module hierarchy comprises 13 RTL source files:

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

An additional utility module (`weight_bram`) provides a 2D weight matrix wrapper with a column-read FSM for future streaming architectures.

### 3.3 Module Descriptions

#### 3.3.1 Processing Element (PE)

The PE is the atomic compute unit — a single multiply-accumulate (MAC) cell. It receives an activation from the left and a weight from the top, computes acc += a * w using full 32-bit precision, and forwards both operands to neighboring PEs. The 32-bit accumulator prevents overflow during long dot-product chains.

Key properties include single-cycle MAC latency, registered inputs and outputs for timing closure, synchronous clear for accumulator reset between tiles, and parameterized data widths via the transformer_pkg.

#### 3.3.2 Systolic Array

A 4×4 grid of PEs forms the systolic matrix multiplication engine. Data flows through the array in a wave pattern: activations propagate left-to-right and weights top-to-bottom. After ROWS + COLS - 1 cycles of streaming, the 16 accumulators hold the complete output tile. This architecture achieves 16 MACs per cycle with only 8 input operands (4 per edge), giving a 2× data reuse factor.

The systolic approach has several advantages for transformer inference: high compute density per unit area, predictable latency (data-independent timing), natural pipelining with no control overhead, and scalability by increasing the array dimensions.

#### 3.3.3 Softmax Unit (Division-Free)

Softmax normalisation computes probs[i] = exp(scores[i] - max) / Σexp. The implementation proceeds through six FSM stages: find maximum score for numerical stability (S_FIND_MAX), subtract maximum and apply PWL exponential approximation (S_SUBTRACT_EXP), accumulate the sum of exponentials (S_SUM), compute the reciprocal of the sum (S_RECIP), and normalise by multiplying each exponential by the reciprocal (S_NORMALIZE).

The PWL exponential uses four linear segments covering [-8, 0] with non-negative clamping at segment boundaries to prevent sign errors. The segments are: near-zero region (-1,0] approximated as 1+x, moderate region (-2,-1] with slope 0.5, steep region (-4,-2] with slope 0.25, and saturation region below -4 clamped to 0.015 (0x0004 in Q8.8).

The normalisation step replaces hardware division with a reciprocal LUT and Newton-Raphson refinement. A 32-entry lookup table (indexed by the top 5 mantissa bits of the CLZ-normalised sum) provides an initial reciprocal estimate in Q2.14. One Newton-Raphson iteration (r₁ = r₀ × (2 − sum × r₀)) doubles precision to approximately 12 bits. A denormalising shift converts the result to the format expected by the Q8.8 multiplier. The entire reciprocal computation is combinational (single-cycle) and uses only a 512-bit ROM, two 16×16 multipliers, and a barrel shifter. Maximum error versus exact division is ±1 LSB (0.0039).

#### 3.3.4 Layer Normalization

LayerNorm computes the mean and variance of the input vector, then normalizes each element. The computation proceeds through three FSM stages: mean computation via sequential accumulation and division, variance computation using centered differences, and element-wise normalization with learnable gamma (scale) and beta (shift) parameters.

The reciprocal square root (1/√(variance + ε)) uses a lookup table defined in transformer_pkg. For production implementations, this would be replaced with a Newton-Raphson iterative refinement stage or a finer-grained LUT.

#### 3.3.5 Multi-Head Attention

The attention module implements the full multi-head scaled dot-product attention with integrated softmax and KV-cache support. A single softmax_unit instance (VEC_LEN = MAX_SEQ_LEN = 128) is time-multiplexed across all N_HEADS = 4 attention heads.

Its operation proceeds through several stages. First, it projects the input to Q, K, V vectors via matrix multiplication with weight matrices Wq, Wk, Wv. Second, it writes K and V to the cache at the current sequence position. Third, it computes attention scores as Q · Kᵀ / √d_k for each head independently. Fourth, it pads future positions with −8.0 (causal mask) and runs softmax per head. Fifth, it computes the probability-weighted sum of V vectors. Finally, it concatenates all heads and applies the output projection Wo.

#### 3.3.6 Feed-Forward Network

The FFN implements a standard two-layer MLP: hidden = ReLU(x · W1 + b1), then output = hidden · W2 + b2. The inner dimension D_FF = 256 is 4× the model dimension, following the standard transformer ratio. ReLU was chosen over GELU for hardware simplicity.

#### 3.3.7 BRAM Modules

Four generic BRAM modules provide synthesizable on-chip memory:

**bram_sp** (single-port) and **bram_dp** (true dual-port) are parameterised by DATA_WIDTH, DEPTH, and INIT_FILE (hex path for $readmemh preloading). Both use synchronous read-first behaviour with 1-cycle read latency and synthesise to Xilinx BRAM36 / Intel M20K primitives.

**kv_cache_bram** wraps a dual-port BRAM for a MAX_SEQ_LEN × D_MODEL cache matrix. Port A provides a vector-write interface: assert wr_start and the module auto-increments through all D_MODEL dimensions. Port B provides element-level random-access reads at (position, dimension) with 1-cycle latency. Address mapping is row-major: mem[position × D_MODEL + dim].

**weight_bram** wraps a single-port BRAM for a ROWS × COLS weight matrix. It provides a column-read FSM that fetches an entire column (ROWS elements) into a registered output buffer, matching the compute modules' access pattern.

#### 3.3.8 Transformer Decoder Top (BRAM-Backed)

The synthesis top-level instantiates 12 single-port weight BRAMs, 2 dual-port KV-cache BRAMs, and the decoder compute core. It provides two weight-loading mechanisms: hex files via INIT_FILE parameters (loaded at synthesis/simulation start), and a runtime weight-loading bus with a unified 16-bit address space mapping all 49,344 parameters across Wq, Wk, Wv, Wo, W1, W2, layer-norm gamma/beta, and FFN biases.

Write-through logic keeps internal register arrays synchronised with the BRAMs, bridging the BRAM storage to the decoder core's array-port interface. A future streaming architecture would replace these register arrays with direct BRAM-to-datapath address/data interfaces.


## 4. Fixed-Point Number System

### 4.1 Q8.8 Format

The design uses signed Q8.8 fixed-point throughout: 1 sign bit, 7 integer bits, and 8 fractional bits in a 16-bit word. This provides a representable range of -128.0 to +127.99609375 with a resolution (LSB) of 1/256 = 0.00390625.

### 4.2 Arithmetic Operations

Multiplication of two Q8.8 values produces a 32-bit result in Q16.16 format. This is right-shifted by 8 (FRAC_BITS) to return to Q8.8. The shift is arithmetic (sign-preserving). Accumulation uses full 32-bit precision to prevent intermediate overflow during dot products of length up to D_MODEL = 64. The final truncation to Q8.8 occurs only at the output.

Saturating addition clamps results to the Q8.8 representable range rather than wrapping. This prevents catastrophic errors from overflow, which is especially important after residual connections where values can grow.

### 4.3 Softmax Reciprocal Arithmetic

The softmax normalisation uses a multi-precision approach. The reciprocal LUT stores entries in Q2.14 format (16,384 = 1.0), providing approximately 6 bits of initial accuracy. The Newton-Raphson refinement operates in Q2.14 internally, with the multiplication r₀ × (2.0 − sum_norm × r₀) computed in 32-bit intermediate precision. The final denormalising shift produces a 16-bit value representing 65536/exp_sum, which when used as the second operand of fp_mul yields the correct Q8.8 normalised probability.

### 4.4 Precision Analysis

For a model dimension of 64 and weights initialized near identity (scale ~0.25-0.5), the Q8.8 format provides approximately 48 dB of signal-to-quantization-noise ratio (SQNR). The softmax reciprocal achieves approximately 12 bits of precision after Newton-Raphson, exceeding the Q8.8 output precision. Scaling to larger models would benefit from Q4.12 or Q8.24 formats for improved fractional precision.


## 5. Verification Strategy

### 5.1 Overview

Verification follows a bottom-up strategy with three complementary layers: a bit-accurate Python behavioural model (54 tests), structural RTL lint (13 files), and iverilog RTL simulation (5 testbenches, 26 tests). All 80 tests pass.

### 5.2 Testbench Hierarchy

**Unit-level SystemVerilog testbenches** cover the Processing Element (6 tests: MAC correctness, forwarding, clear, negative numbers), Systolic Array (4 tests: identity multiply, done signal, clear-and-reuse, uniform input), Softmax Unit (4 tests: uniform inputs, dominant score, negative scores, sum check), and BRAM modules (8 tests: single-port write/read, sequential access, hex init simulation, dual-port cross-read, simultaneous reads, dual-port init).

**Integration-level SystemVerilog testbench** covers the full Transformer Decoder (4 tests: single-token inference at position 0 with non-zero output verification, KV-cache write verification, and sequential two-token inference).

**CocoTB testbenches** provide Python-based verification with randomized testing for the Processing Element and Softmax Unit.

**Bit-accurate behavioural model** (`verify_behavioral.py`, 54 tests) mirrors the RTL at the bit level using identical Q8.8 fixed-point arithmetic, accumulator widths, reciprocal-LUT logic, and FSM sequencing. This model serves as both a golden reference and a standalone verification environment requiring no HDL simulator.

### 5.3 Packed Array Port Strategy

Icarus Verilog 12.0 does not propagate values through SystemVerilog unpacked array ports (e.g. `data_t out [N]`). This was resolved by converting all 1D vector ports to packed arrays (`logic signed [N-1:0][DATA_WIDTH-1:0]`), which iverilog handles correctly as contiguous bit vectors. Multi-dimensional weight matrices remain as unpacked arrays since they are initialised before simulation and do not require runtime port propagation. This strategy achieves full simulation correctness in iverilog while maintaining synthesizability for commercial tools.

### 5.4 Behavioural Verification Results

The full behavioural verification suite (54 tests) passes with the following breakdown:

Fixed-Point Utilities (16/16 passed): Q8.8 roundtrip conversion for edge values (0.0, 127.0, -128.0, LSB), multiply correctness for positive, negative, and fractional operands, saturating addition with positive and negative overflow.

Processing Element (9/9 passed): Reset clears accumulator, MAC 2.0×3.0 = 6.0 in full precision (0x60000), 4-cycle accumulation (4 × 1.0×1.0 = 262144), data forwarding a_out/w_out, clear mid-operation, negative multiply -2.0×3.0 and -1.5×2.0, 20-operation randomized MAC vs golden model (seed=42, exact match).

Systolic Array (8/8 passed): Single-element [0][0] = 1.0×2.0 (0x20000), done signal assertion, clear-and-reuse cycle, full 2×2 matrix multiply A=[[1,2],[3,4]] B=[[5,6],[7,8]] producing C=[[19,22],[43,50]] with proper systolic wave staggering.

Softmax Unit (8/8 passed): Uniform inputs produce exactly equal 0.125 probabilities, sum = 1.0000, dominant score (4.0 vs 0.0) yields 0.8984 probability, monotonic ordering preserved for ramp inputs, less-negative scores get higher probability, all-zero uniformity, back-to-back execution with no state leakage. These tests validate the reciprocal-LUT normalisation matching exact division to within ±1 LSB.

Layer Normalization (5/5 passed): Constant input normalizes to zero, symmetric ±1.0 preserves sign with correct scaling, gamma=2.0 doubles output magnitude (ratio=2.00), beta=1.0 offsets zero-input output to 1.0, ramp input centres output with mean=0.0000.

Feed-Forward Network (4/4 passed): ReLU zeros all negative pre-activations, positive inputs pass through correctly, negative-only input produces all-zero hidden layer, zero input propagates bias terms.

Decoder Integration (12/12 passed): Full pipeline (LN1 → Attention projection → softmax with causal mask → Residual 1 → LN2 → FFN → Residual 2) produces non-zero output, KV-cache write verified, single-position probability ≈ 0.90, multi-position sum ≈ 0.95, future-position masking < 0.01, output energy (4.91) exceeds input energy (2.00) confirming propagation through all stages, second token at position 1 processes correctly.

### 5.5 RTL Simulation Results

All 26 RTL simulation tests pass across 5 testbenches:

Processing Element: 6/6 passed. MAC correctness, forwarding, accumulation, clear, and negative values all verified with correct numerical output.

Systolic Array: 4/4 passed. Result matrix values fully visible and correct (e.g. result[0] = 0x00020000 = 2.0 in Q16.16 accumulator). Done signal, clear-and-reuse, and uniform input all verified.

Softmax Unit: 4/4 passed. Uniform inputs produce 0x0020 (0.125) for all elements. Dominant score (4.0) produces 0x00E6 (0.898). Negative scores produce monotonically decreasing probabilities.

BRAM: 8/8 passed. Single-port write/read, sequential 8-value write/readback, simulated hex init, dual-port cross-port access, simultaneous reads from both ports, and dual-port hex init.

Decoder Integration: 4/4 passed. Single-token inference produces non-zero output (e.g. out_emb[0] = 0x00A8 = 0.656). KV-cache write mechanism verified. Sequential two-token inference completes successfully.

Additionally, the full 13-file hierarchy including `transformer_decoder_top` with all BRAM modules compiles cleanly with no errors.

### 5.6 RTL Lint Results

A structural lint pass over all 13 RTL files verified: balanced module/endmodule and package/endpackage pairs, proper always_ff blocks with reset in sensitivity lists (5 warnings for BRAM blocks that intentionally omit reset — standard practice for FPGA BRAM inference), correct transformer_pkg import in all modules, and cross-file instantiation resolution.

### 5.7 Coverage

The combined test suite covers: basic functional correctness of each sub-module, bit-exact fixed-point arithmetic across all operations, data forwarding through the systolic array with correct wave propagation, FSM state transitions including back-to-back operation, fixed-point edge cases (negative numbers, saturation, zero inputs), reciprocal-LUT softmax normalisation accuracy, BRAM read/write timing and initialization, integration across the full decoder pipeline, KV-cache write mechanics, sequential multi-token inference, and correct value propagation through all module port boundaries.


## 6. Implementation Metrics (Estimated)

The following estimates assume a Xilinx Artix-7 FPGA target at the default parameters.

For the Processing Element: 1 DSP48 slice, approximately 30 FFs, and approximately 20 LUTs.

For the 4×4 Systolic Array: 16 DSP48 slices, approximately 500 FFs, and approximately 350 LUTs.

For the Softmax Unit (VEC_LEN=128): 2 DSP48 (for Newton-Raphson multiplies), approximately 2K FFs, and approximately 3K LUTs including the 32-entry reciprocal LUT.

For the Layer Normalization: 2-4 DSP48, approximately 300 FFs, and approximately 500 LUTs.

For the Multi-Head Attention: approximately 64 DSP48 (dominated by projections), approximately 5K FFs, and approximately 8K LUTs.

For the Feed-Forward Network: approximately 64 DSP48, approximately 3K FFs, and approximately 5K LUTs.

For the Weight BRAMs (12 instances): approximately 110 BRAM18K (49,344 × 16-bit words ≈ 96 KB).

For the KV-Cache BRAMs (2 instances): approximately 16 BRAM18K (16,384 × 16-bit words ≈ 32 KB).

For the full Transformer Decoder Top: approximately 150 DSP48, approximately 11K FFs, approximately 17K LUTs, and approximately 126 BRAM18K.

Clock frequency estimate: 100-200 MHz depending on place-and-route and constraint effort.


## 7. Limitations and Future Work

### 7.1 Resolved Limitations

Several limitations from earlier revisions have been addressed. BRAM-backed weight storage replaces the original combinational weight arrays, with 12 single-port BRAMs holding all parameters and a runtime weight-loading bus for dynamic updates. The softmax normalisation now uses a division-free reciprocal-LUT with Newton-Raphson refinement, replacing the non-synthesizable Verilog `/` operator. The softmax_unit is fully integrated into multi_head_attention, time-multiplexed across all attention heads. The iverilog simulation limitation (unpacked array ports showing `xxxx`) has been resolved by converting 1D vector ports to packed arrays, achieving 80/80 tests passing.

### 7.2 Current Limitations

The design still has several limitations. The bridge architecture in `transformer_decoder_top` uses register arrays to feed the decoder core's array ports from BRAM contents, which adds area overhead. The reciprocal square root in LayerNorm uses a coarse 4-entry LUT. Only ReLU activation is supported, not GELU/SiLU. The design processes one token at a time with no batching support.

### 7.3 Planned Enhancements

Future development could address: a streaming weight interface that eliminates the register-array bridge by generating BRAM addresses directly from the decoder FSM state, a finer-grained LayerNorm reciprocal sqrt using the same LUT+Newton-Raphson approach proven in the softmax, GeLU/SiLU activation support via PWL approximation, parallel softmax with N_HEADS instances for 4× lower latency, tiled execution for larger model dimensions exceeding the systolic array size, multi-layer stacking with a top-level sequencer, int8/int4 quantization modes for improved density, and AXI-Lite control/status register interface for SoC integration.


## 8. Conclusion

This project demonstrates a complete, synthesizable transformer decoder block in SystemVerilog, suitable for FPGA prototyping and as a reference architecture for LLM inference accelerator design. The 13-module hierarchy — spanning processing elements, systolic array, division-free softmax, layer normalisation, multi-head attention, feed-forward network, BRAM storage, and a BRAM-backed top level — mirrors the conceptual structure of the transformer while addressing practical synthesis concerns.

The fixed-point arithmetic, systolic dataflow, reciprocal-LUT softmax, and BRAM-backed KV-cache architecture represent the core design patterns used in production LLM inference hardware. All 80 verification tests pass (54 behavioural, 26 RTL simulation) across the full hierarchy, with correct numerical output visible at every module boundary. While the current parameter scale is small (D_MODEL=64), the architecture scales to production dimensions by increasing array sizes and adding memory hierarchy.
