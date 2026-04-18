# inferlib — Fast LLM Inference Primitives in Rust

## What it is
A PyO3 library of Rust-accelerated LLM inference primitives. Drop into any Python inference stack (vLLM, TGI, llama.cpp Python bindings, HF Transformers, text-generation-webui, etc.) as `pip install inferlib`. No Rust knowledge required to use.

**First product:** Logit sampling kernels — top-k, top-p/nucleus, min-p, typical sampling, and repetition penalty. 10-100x faster than pure Python/NumPy equivalents.

**Later:** Constrained decoding / grammar masking, speculative verification, KV cache ops.

## Why this wins
LLM inference stacks are full of pure-Python bottlenecks that everyone works around with CUDA kernels or workaround libraries. Nobody's made a clean, fast, drop-in Rust replacement for the most common ones. The market is wide open.

## Product structure

### Tier 1 — Open core (free, PyPI)
- Rust CPU kernels for all samplers
- MIT license
- Ship on PyPI as `inferlib`
- Builds via `maturin` — one `pip install` no Rust toolchain needed

### Tier 2 — GPU extensions (paid, ~$50-500/mo)
- CUDA kernels (via `candle-core` or raw CUDA) for same ops
- Batch inference primitives
- Faster dequantization paths

### Tier 3 — Enterprise (custom contracts)
- License to companies running inference infrastructure
- Custom kernels for their hardware
- Support SLAs

## GitHub setup
- **Org:** `inferlib-ai` (or similar, claim before someone else does)
- **Main repo:** `inferlib` (private initially, open after Tier 1 ships)
- **Lang:** Rust + Python bindings via PyO3/maturin
- **CI:** GitHub Actions — test Linux/Mac/Windows, publish to PyPI on tag

## Technical design

### Crate structure
```
inferlib/
├── inferlib-python/     # PyO3 bindings (maturin project)
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── pyproject.toml
├── inferlib-core/       # Core Rust library
│   ├── src/
│   │   ├── lib.rs
│   │   ├── sampling.rs   # All sampler kernels
│   │   └── ffi.rs       # C-compatible FFI for non-Python callers
│   └── Cargo.toml
└── benches/             # criterion benchmarks
```

### Sampling kernel API (Python)
```python
import inferlib

# Top-k sampling
next_token_ids = inferlib.top_k(logits, k=50)

# Nucleus (top-p) sampling
next_token_ids = inferlib.top_p(logits, p=0.9)

# Min-p sampling
next_token_ids = inferlib.min_p(logits, p=0.1)

# Typical sampling
next_token_ids = inferlib.typical(logits, mass=0.9)

# Repetition penalty
penalized = inferlib.repetition_penalty(logits, input_ids, penalty=1.1)

# Combined (what inference servers actually do)
next_token = inferlib.sample(
    logits,
    method="nucleus",  # top_k | top_p | min_p | typical | greedy
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    input_ids=input_ids,  # for repetition penalty
)
```

### Rust core trait
```rust
pub trait Sampler {
    fn sample(&self, logits: &mut [f32], input_ids: Option<&[u32]>) -> u32;
}

pub struct NucleusSampler {
    pub p: f32,
    pub temperature: f32,
}
pub struct TopKSampler { pub k: usize, pub temperature: f32 }
pub struct MinPSampler { pub p: f32, pub temperature: f32 }
pub struct RepetitionPenaltySampler { pub penalty: f32 }
```

### Benchmark target
- Pure Python/NumPy equivalent (baseline)
- `torch.multinomial` + custom ops
- `transformers.generation.utils` samplers
- Compete on: latency per token (ms), throughput (tokens/sec), memory allocation

## Name check
- `inferlib` — inference library — clear purpose
- Not taken on PyPI (verify)
- Not taken on GitHub (verify)
- `inferlib-ai` org on GitHub (verify)

## Action items
1. Verify name availability (PyPI + GitHub)
2. Create GitHub org `inferlib-ai`
3. Create repo + push initial structure
4. Write Rust sampling kernels (top_k, top_p, min_p, typical, repetition_penalty)
5. PyO3 bindings via maturin
6. CI/CD: test + publish to TestPyPI → PyPI
7. Ship v0.1.0 to PyPI
8. Twitter / HN launch

## Timeline
- Day 1: Name, org, repo, initial structure
- Day 2-3: Rust sampling kernels implemented + benchmarked
- Day 4: PyO3 bindings + maturin setup
- Day 5: CI/CD + first PyPI release
- Week 2: Add constrained decoding
- Month 1: Evaluate paid GPU tier

## Monetization plan
1. **Free PyPI package** → builds reputation, users, credibility
2. **GitHub Sponsors / Ko-fi** → voluntary donations from users
3. **GPU extension paid tier** → actual revenue
4. **B2B licensing** → bigger contracts

First dollar: GitHub Sponsors the moment the package is usable and someone finds it valuable.
