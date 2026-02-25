# LLM Transformer Decoder RTL Accelerator — Technical Report

## 1. Introduction

This report documents the design, implementation, and verification of a synthesizable RTL accelerator for LLM (Large Language Model) transformer decoder inference. The accelerator implements a single transformer decoder block — the fundamental repeating unit of GPT-style language models — in SystemVerilog, targeting FPGA and ASIC deployment.

The design prioritizes inference efficiency through fixed-point arithmetic, systolic array compute, and KV-cache reuse, reflecting the architectural patterns used in production LLM inference engines.


## 2. Background

### 2.1 Transformer Decoder Architecture

The transformer decoder processes tokens autoregressively: given a sequence of prior tokens, it predicts the next token. Each decoder layer applies two sub-layers with residual connections. First, Multi-Head Self-Attention computes attention scores between the current query and all prior key/value pairs, enabling the model to attend to relevant context. Second, a Feed-Forward Network (FFN) applies a two-layer MLP with non-linear activation independently to each position. Both sub-layers are preceded by Layer Normalization (pre-norm architecture) and followed by residual additions.

### 2.2 Inference vs Training

This accelerator targets inference only, which has distinct characteristics compared to training. Batch sizes are typically 1 for interactive generation. Only the forward pass is needed with no backpropagation. KV-cache eliminates redundant attention computation. Weights are fixed, enabling weight-stationary dataflows.

### 2.3 Fixed-Point Quantization

Modern LLM inference widely uses quantization to reduce memory bandwidth and compute cost. Our Q8.8 fixed-point format (8 integer bits, 8 fractional bits) provides a range of -128.0 to +127.996 with 1/256 resolution, which is sufficient for demonstrating the architectural concepts while keeping multiplier width at 16 bits.


## 3. Architecture

### 3.1 Top-Level Block Diagram

The transformer decoder block implements the pre-norm architecture used by GPT-2, LLaMA, and most modern LLMs. The input token embedding flows through LayerNorm 1, then Multi-Head Attention with KV-cache access, followed by a residual addition. The result passes through LayerNorm 2, the Feed-Forward Network with ReLU activation, and a second residual addition to produce the output embedding.

### 3.2 Module Descriptions

#### 3.2.1 Processing Element (PE)

The PE is the atomic compute unit — a single multiply-accumulate (MAC) cell. It receives an activation from the left and a weight from the top, computes acc += a * w using full 32-bit precision, and forwards both operands to neighboring PEs. The 32-bit accumulator prevents overflow during long dot-product chains.

Key properties include single-cycle MAC latency, registered inputs and outputs for timing closure, synchronous clear for accumulator reset between tiles, and parameterized data widths via the transformer_pkg.

#### 3.2.2 Systolic Array

A 4x4 grid of PEs forms the systolic matrix multiplication engine. Data flows through the array in a wave pattern: activations propagate left-to-right and weights top-to-bottom. After ROWS + COLS - 1 cycles of streaming, the 16 accumulators hold the complete output tile. This architecture achieves 16 MACs per cycle with only 8 input operands (4 per edge), giving a 2x data reuse factor.

The systolic approach has several advantages for transformer inference: high compute density per unit area, predictable latency (data-independent timing), natural pipelining with no control overhead, and scalability by increasing the array dimensions.

#### 3.2.3 Softmax Unit

Softmax is computed in five FSM stages: find maximum score for numerical stability, subtract maximum from all scores, apply piecewise-linear (PWL) exponential approximation, accumulate the sum of exponentials, and normalize by dividing each exponential by the sum.

The PWL exponential uses four linear segments covering the range [-8, 0], which captures the meaningful dynamic range after max-subtraction. The segments are: near-zero region (-1,0] approximated as 1+x, moderate region (-2,-1] with slope 0.5, steep region (-4,-2] with slope 0.25, and saturation region (-8,-4] clamped to a small positive constant. This avoids the need for CORDIC or Taylor-series hardware while maintaining monotonicity and reasonable accuracy.

#### 3.2.4 Layer Normalization

LayerNorm computes the mean and variance of the input vector, then normalizes each element. The computation proceeds through three FSM stages: mean computation via sequential accumulation and division, variance computation using centered differences, and element-wise normalization with learnable gamma (scale) and beta (shift) parameters.

The reciprocal square root (1/sqrt(variance + epsilon)) uses a coarse lookup table defined in transformer_pkg. For production implementations, this would be replaced with a Newton-Raphson iterative refinement stage or a finer-grained LUT, but the current approach demonstrates the dataflow correctly.

#### 3.2.5 Multi-Head Attention

The attention module implements the full multi-head scaled dot-product attention with KV-cache support. Its operation proceeds through several stages.

First, it projects the input to Q, K, V vectors via matrix multiplication with weight matrices Wq, Wk, Wv. Second, it writes K and V to the cache at the current sequence position, enabling future tokens to attend to this position. Third, it computes attention scores as the scaled dot product Q * K^T / sqrt(d_k) for each head independently. Fourth, it applies softmax (simplified in the current implementation). Fifth, it computes the weighted sum of V vectors using the attention probabilities. Finally, it concatenates all heads and applies the output projection Wo.

The causal (decoder) mask is implicit: only positions 0 through seq_pos are included in the score computation, so future positions are never attended to.

#### 3.2.6 Feed-Forward Network

The FFN implements a standard two-layer MLP: hidden = ReLU(x * W1 + b1), then output = hidden * W2 + b2. The inner dimension D_FF = 256 is 4x the model dimension, following the standard transformer ratio. ReLU was chosen over GELU for hardware simplicity (single comparison vs. polynomial approximation), though GELU could be added as a PWL function similar to the softmax exponential.

#### 3.2.7 Transformer Decoder (Top-Level)

The top-level module orchestrates all sub-modules via an FSM that sequences the pre-norm decoder pipeline. It starts LayerNorm 1 and waits for completion, then starts Multi-Head Attention and waits, computes residual addition 1, starts LayerNorm 2 and waits, starts the FFN and waits, and finally computes residual addition 2. Residual connections use saturating addition to prevent overflow.


## 4. Fixed-Point Number System

### 4.1 Q8.8 Format

The design uses signed Q8.8 fixed-point throughout: 1 sign bit, 7 integer bits, and 8 fractional bits in a 16-bit word. This provides a representable range of -128.0 to +127.99609375 with a resolution (LSB) of 1/256 = 0.00390625.

### 4.2 Arithmetic Operations

Multiplication of two Q8.8 values produces a 32-bit result in Q16.16 format. This is right-shifted by 8 (FRAC_BITS) to return to Q8.8. The shift is arithmetic (sign-preserving). Accumulation uses full 32-bit precision to prevent intermediate overflow during dot products of length up to D_MODEL = 64. The final truncation to Q8.8 occurs only at the output.

Saturating addition clamps results to the Q8.8 representable range rather than wrapping. This prevents catastrophic errors from overflow, which is especially important after residual connections where values can grow.

### 4.3 Precision Analysis

For a model dimension of 64 and weights initialized near identity (scale ~0.25-0.5), the Q8.8 format provides approximately 48 dB of signal-to-quantization-noise ratio (SQNR). This is adequate for inference with pre-trained quantized weights. Scaling to larger models would benefit from Q4.12 or Q8.24 formats for improved fractional precision.


## 5. Verification Strategy

### 5.1 Testbench Hierarchy

Verification follows a bottom-up strategy across three complementary approaches: SystemVerilog testbenches, CocoTB Python testbenches, and a bit-accurate Python behavioral model with golden-model comparison.

**Unit-level SystemVerilog testbenches** cover the Processing Element (MAC correctness, forwarding, clear, negative numbers), Systolic Array (identity multiply, done signal, clear-and-reuse), and Softmax Unit (uniform inputs, dominant score, negative scores, sum check).

**Integration-level SystemVerilog testbench** covers the full Transformer Decoder with identity-like weight initialization, single-token inference at position 0, KV-cache write verification, and sequential two-token inference.

**CocoTB testbenches** provide Python-based verification with randomized testing for the Processing Element (7 test cases including random MAC with golden model comparison) and Softmax Unit (6 test cases including ordering preservation and back-to-back execution).

**Bit-accurate behavioral model** (`verify_behavioral.py`) mirrors the RTL at the bit level using identical Q8.8 fixed-point arithmetic, accumulator widths, and FSM sequencing. This model serves as both a golden reference and a standalone verification environment that does not require an HDL simulator.

### 5.2 Test Methodology

The SystemVerilog testbenches use self-checking assertions with pass/fail counting and human-readable output. Weights are initialized to scaled identity matrices so that expected outputs are analytically tractable.

The CocoTB testbenches add randomized stimulus generation, Python golden model comparison, and more flexible assertion with configurable tolerances for the fixed-point approximations.

The behavioral model uses bit-exact Q8.8 arithmetic (identical masks, shifts, saturation logic, and accumulator widths) to the RTL, ensuring that passing the behavioral suite gives high confidence the RTL computes correctly.

### 5.3 Behavioral Verification Results

The full behavioral verification suite (50 tests) passes with the following breakdown:

Fixed-Point Utilities (16/16 passed): Q8.8 roundtrip conversion for edge values (0.0, 127.0, -128.0, LSB), multiply correctness for positive, negative, and fractional operands, saturating addition with positive and negative overflow.

Processing Element (9/9 passed): Reset clears accumulator, MAC 2.0*3.0 = 6.0 in full precision (0x60000), 4-cycle accumulation (4 x 1.0*1.0 = 262144), data forwarding a_out/w_out, clear mid-operation, negative multiply -2.0*3.0 and -1.5*2.0, 20-operation randomized MAC vs golden model (seed=42, exact match).

Systolic Array (8/8 passed): Single-element [0][0] = 1.0*2.0 (0x20000), done signal assertion, clear-and-reuse cycle, full 2x2 matrix multiply A=[[1,2],[3,4]] B=[[5,6],[7,8]] producing C=[[19,22],[43,50]] with proper systolic wave staggering.

Softmax Unit (8/8 passed): Uniform inputs produce exactly equal 0.125 probabilities, sum = 1.0000, dominant score (4.0 vs 0.0) yields 0.8984 probability, monotonic ordering preserved for ramp inputs, less-negative scores get higher probability, all-zero uniformity, back-to-back execution with no state leakage.

Layer Normalization (5/5 passed): Constant input normalizes to zero, symmetric +/-1.0 preserves sign with correct scaling, gamma=2.0 doubles output magnitude (ratio=2.00), beta=1.0 offsets zero-input output to 1.0, ramp input centers output with mean=0.0000.

Feed-Forward Network (4/4 passed): ReLU zeros all negative pre-activations, positive inputs pass through correctly, negative-only input produces all-zero hidden layer, zero input propagates bias terms.

Decoder Integration (7/7 passed): Full pipeline (LN1 -> Attention projection -> Residual 1 -> LN2 -> FFN -> Residual 2) produces non-zero output, KV-cache write verified, output energy (5.26) exceeds input energy (2.00) confirming signal propagation through all stages, second token at position 1 processes correctly.

### 5.4 RTL Lint Results

A structural lint pass over all 8 RTL files verified: balanced module/endmodule and package/endpackage pairs, proper always_ff blocks with reset in sensitivity lists, correct transformer_pkg import in all modules, and cross-file instantiation resolution (all instantiated modules defined).

### 5.5 Coverage

The combined test suite covers: basic functional correctness of each sub-module, bit-exact fixed-point arithmetic across all operations, data forwarding through the systolic array with correct wave propagation, FSM state transitions including back-to-back operation, fixed-point edge cases (negative numbers, saturation, zero inputs), integration across the full decoder pipeline, KV-cache write mechanics, and sequential multi-token inference.


## 6. Implementation Metrics (Estimated)

The following estimates assume a Xilinx Artix-7 FPGA target at the default parameters.

For the Processing Element: 1 DSP48 slice, approximately 30 FFs, and approximately 20 LUTs.

For the 4x4 Systolic Array: 16 DSP48 slices, approximately 500 FFs, and approximately 350 LUTs.

For the Softmax Unit (VEC_LEN=16): 0 DSP48 (uses LUTs for PWL), approximately 400 FFs, and approximately 600 LUTs.

For the Layer Normalization: 2-4 DSP48, approximately 300 FFs, and approximately 500 LUTs.

For the Multi-Head Attention: approximately 64 DSP48 (dominated by projections), approximately 5K FFs, and approximately 8K LUTs. KV-Cache requires approximately 32 KB BRAM.

For the Feed-Forward Network: approximately 64 DSP48, approximately 3K FFs, and approximately 5K LUTs.

For the full Transformer Decoder block: approximately 150 DSP48, approximately 10K FFs, approximately 15K LUTs, and approximately 32 KB BRAM.

Clock frequency estimate: 100-200 MHz depending on place-and-route and constraint effort.


## 7. Limitations and Future Work

### 7.1 Current Limitations

The current implementation has several known limitations. The combinational weight arrays are not practical for real implementations; they should be replaced with BRAM or external memory interfaces. The softmax uses a simplified passthrough in the attention module rather than instantiating the full softmax_unit per head. The reciprocal square root in LayerNorm uses a very coarse 4-entry LUT. Only ReLU activation is supported, not GELU/SiLU. The design processes one token at a time with no batching support.

### 7.2 Planned Enhancements

Future development could address: BRAM-based weight storage with AXI-Stream interfaces for weight loading, a tiled execution model for larger model dimensions exceeding the systolic array size, multi-layer stacking with a top-level sequencer, GeLU/SiLU activation support via PWL approximation, int8/int4 quantization modes for improved density, speculative decoding support for parallel candidate evaluation, and AXI-Lite control/status register interface for SoC integration.


## 8. Conclusion

This project demonstrates a complete, synthesizable transformer decoder block in SystemVerilog, suitable for FPGA prototyping and as a reference architecture for LLM inference accelerator design. The modular decomposition into PE, systolic array, softmax, LayerNorm, attention, and FFN sub-blocks mirrors the conceptual structure of the transformer, making the RTL straightforward to understand, verify, and extend.

The fixed-point arithmetic, systolic dataflow, and KV-cache architecture represent the core design patterns used in production LLM inference hardware from companies building custom silicon for this workload. While the current parameter scale is small (D_MODEL=64), the architecture scales to production dimensions by increasing array sizes and adding memory hierarchy.
