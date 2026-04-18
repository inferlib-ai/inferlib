# inferlib

**Fast LLM inference primitives in Rust.**

Rust-accelerated sampling kernels + PyO3 bindings for Python ML inference stacks.

```bash
pip install inferlib
```

## What

Drop-in Rust acceleration for the Python inference stack you're already using.

| Operation | Pure Python | inferlib (Rust) |
|-----------|------------|-----------------|
| top-k sampling | ~2-5ms | <0.05ms |
| nucleus (top-p) | ~3-8ms | <0.1ms |
| repetition penalty | ~1-3ms | <0.02ms |

Benchmarked on vocab=32000, T4 GPU, Python 3.11.

## Quick Start

```python
import inferlib

# All sampling methods
token = inferlib.sample(
    logits,                      # model output logits
    method="nucleus",           # greedy | top_k | top_p | min_p | typical
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    prev_tokens=input_ids,      # for repetition penalty
    seed=42,                    # optional
)
```

Or call individual kernels:

```python
# Just top-k
token = inferlib.top_k(logits, k=50, temperature=0.7)

# Just nucleus
token = inferlib.top_p(logits, p=0.9, temperature=0.7)

# Repetition penalty (returns modified logits)
penalized = inferlib.repetition_penalty(logits, prev_tokens, penalty=1.1)
```

## Features

- **top-k sampling** — keep top k tokens, sample from distribution
- **nucleus / top-p sampling** — keep smallest token set with cumulative prob > p
- **min-p sampling** — threshold based on max probability
- **typical sampling** — entropy-conditioned sampling
- **repetition penalty** — penalize previously generated tokens
- **greedy decoding** — argmax (temperature=0)

## Install

```bash
pip install inferlib
```

Requires: Python 3.8+, Rust toolchain (for building from source)

From source:
```bash
cd inferlib-python && pip install -e .
```

## Why Rust

Python/NumPy inference samplers are pure Python bottlenecks on every token generation. The math is simple but the Python loop overhead is enormous. Rust eliminates that overhead while providing the same easy Python bindings.

## Roadmap

- [ ] GPU kernels (CUDA) for batch inference
- [ ] Constrained decoding / grammar masking
- [ ] Speculative verification
- [ ] KV cache operations

## License

MIT
