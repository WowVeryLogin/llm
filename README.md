# llm

Simple LLM implemented purely in Rust without any specific ML libraries (using only ndarray crate).

Features:
- Manual forward and backward passes
- Multi-Head Self-Attention
- LayerNorm
- AdamW optimizer
- Dropout
- Token and positional embeddings
- Autoregressive text generation
- End-to-end training without ML frameworks

Implemented a transformer from scratch in Rust, including forward pass, backpropagation, AdamW, LayerNorm, self-attention, embeddings, dropout, training loop, and can train it end-to-end.

The goal was to refresh the math behind every tensor and gradient, and also to optimize it enough to train on my laptops CPU.

Architecture:

    Token Embedding
    + Position Embedding
    ↓
    N × Transformer Block
        x = x + MultiHeadSelfAttention(
                    LayerNorm(x),
                    attention_dropout,
                    projection_dropout
                )

        x = x + FeedForward(
                    LayerNorm(x),
                    dropout
                )
    ↓
    Final LayerNorm
    ↓
    LM Head
    ↓
    Cross Entropy Loss

It uses a single char as a token (it is very easy to add more production ready tokenizer).

## How to run

```rust
cargo run --release
```

This will use set.txt file (bible text) as training data and continue it with 500 symbols.

## Future improvements

The goal was to refresh transformer math. Real production requires:

Training:
- GPU/TPU acceleration (CUDA, Metal, Vulkan, etc.)
- Mixed precision (FP16/BF16)
- Better initialization and scheduling strategies

Inference:
- KV-cache
- FlashAttention
- PagedAttention
- Speculative decoding
- Continuous batching
- Quantization (INT8/INT4)

Model:
- BPE tokenizer
- Rotary positional embeddings (RoPE)
- GELU activation
- Weight tying
