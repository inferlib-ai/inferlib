//! PyO3 bindings for inferlib-core

use pyo3::prelude::*;
use inferlib_core::sampling as core;

/// Apply repetition penalty to logits (returns new Vec).
#[pyfunction]
#[pyo3(name = "repetition_penalty")]
pub fn py_repetition_penalty(
    logits: Vec<f32>,
    prev_tokens: Vec<u32>,
    penalty: f32,
) -> Vec<f32> {
    let mut logits = logits;
    core::apply_repetition_penalty(&mut logits, &prev_tokens, penalty);
    logits
}

/// Top-k sampling.
#[pyfunction]
#[pyo3(name = "top_k")]
pub fn py_top_k(mut logits: Vec<f32>, k: usize, temperature: f32, seed: Option<u64>) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    core::sample_top_k(&mut logits, k, temperature, &mut rng)
}

/// Nucleus (top-p) sampling.
#[pyfunction]
#[pyo3(name = "top_p")]
pub fn py_top_p(mut logits: Vec<f32>, p: f32, temperature: f32, seed: Option<u64>) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    core::sample_nucleus(&mut logits, p, temperature, &mut rng)
}

/// Min-p sampling.
#[pyfunction]
#[pyo3(name = "min_p")]
pub fn py_min_p(mut logits: Vec<f32>, min_p: f32, temperature: f32, seed: Option<u64>) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    core::sample_min_p(&mut logits, min_p, temperature, &mut rng)
}

/// Typical sampling.
#[pyfunction]
#[pyo3(name = "typical")]
pub fn py_typical(mut logits: Vec<f32>, mass: f32, temperature: f32, seed: Option<u64>) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    core::sample_typical(&mut logits, mass, temperature, &mut rng)
}

/// Greedy decoding.
#[pyfunction]
#[pyo3(name = "greedy")]
pub fn py_greedy(logits: Vec<f32>) -> u32 {
    core::sample_greedy(&logits)
}

/// Combined sampler.
///
/// Args:
///     logits: Raw model output logits
///     method: "greedy" | "top_k" | "top_p" | "min_p" | "typical"
///     temperature: Sampling temperature (0.0 = greedy)
///     top_p: Nucleus p value
///     top_k: Top-k value
///     min_p: Min-p threshold
///     repetition_penalty: Penalty for repeated tokens
///     prev_tokens: Previously generated tokens
///     seed: Random seed (None = random)
#[pyfunction]
#[pyo3(name = "sample")]
pub fn py_sample(
    mut logits: Vec<f32>,
    method: &str,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    prev_tokens: Option<Vec<u32>>,
    seed: Option<u64>,
) -> u32 {
    core::sample(
        &mut logits,
        method,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        prev_tokens.as_deref(),
        seed,
    )
}

/// Python module definition.
#[pymodule]
pub fn inferlib(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_p, m)?)?;
    m.add_function(wrap_pyfunction!(py_min_p, m)?)?;
    m.add_function(wrap_pyfunction!(py_typical, m)?)?;
    m.add_function(wrap_pyfunction!(py_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample, m)?)?;
    m.add_function(wrap_pyfunction!(py_repetition_penalty, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
