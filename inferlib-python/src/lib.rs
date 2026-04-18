//! PyO3 bindings for inferlib-core

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use inferlib_core::sampling as core;

/// Apply repetition penalty to logits (Python-callable)
#[pyfunction]
#[pyo3(name = "repetition_penalty")]
pub fn py_repetition_penalty(
    logits: Vec<f32>,
    prev_tokens: Vec<u32>,
    penalty: f32,
) -> Vec<f32> {
    let mut logits = logits;
    inferlib_core::sampling::apply_repetition_penalty(&mut logits, &prev_tokens, penalty);
    logits
}

/// Top-k sampling
#[pyfunction]
#[pyo3(name = "top_k")]
pub fn py_top_k(
    mut logits: Vec<f32>,
    k: usize,
    temperature: f32,
    seed: Option<u64>,
) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    inferlib_core::sampling::sample_top_k(&mut logits, k, temperature, &mut rng)
}

/// Nucleus (top-p) sampling
#[pyfunction]
#[pyo3(name = "top_p")]
pub fn py_top_p(
    mut logits: Vec<f32>,
    p: f32,
    temperature: f32,
    seed: Option<u64>,
) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    inferlib_core::sampling::sample_nucleus(&mut logits, p, temperature, &mut rng)
}

/// Min-p sampling
#[pyfunction]
#[pyo3(name = "min_p")]
pub fn py_min_p(
    mut logits: Vec<f32>,
    min_p: f32,
    temperature: f32,
    seed: Option<u64>,
) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    inferlib_core::sampling::sample_min_p(&mut logits, min_p, temperature, &mut rng)
}

/// Typical sampling
#[pyfunction]
#[pyo3(name = "typical")]
pub fn py_typical(
    mut logits: Vec<f32>,
    mass: f32,
    temperature: f32,
    seed: Option<u64>,
) -> u32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    inferlib_core::sampling::sample_typical(&mut logits, mass, temperature, &mut rng)
}

/// Greedy decoding
#[pyfunction]
#[pyo3(name = "greedy")]
pub fn py_greedy(logits: Vec<f32>) -> u32 {
    inferlib_core::sampling::sample_greedy(&logits)
}

/// Combined sampling with all methods
/// 
/// Args:
///     logits: Raw model output logits (will be modified in-place)
///     method: "greedy" | "top_k" | "top_p" | "min_p" | "typical"
///     temperature: Sampling temperature (0.0 = greedy)
///     top_p: Nucleus p value (for top_p method)
///     top_k: Top-k value
///     min_p: Min-p value (for min_p method)
///     repetition_penalty: Penalty for repeated tokens
///     prev_tokens: Previously generated tokens (for repetition penalty)
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
    inferlib_core::sampling::sample(
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

/// Python module definition
pub fn get_module() -> PyResult<PyModule> {
    let m = PyModule::new(crate_name!(), "inferlib")?;
    
    m.add_wrapped(wrap_pyfunction!(py_top_k))?;
    m.add_wrapped(wrap_pyfunction!(py_top_p))?;
    m.add_wrapped(wrap_pyfunction!(py_min_p))?;
    m.add_wrapped(wrap_pyfunction!(py_typical))?;
    m.add_wrapped(wrap_pyfunction!(py_greedy))?;
    m.add_wrapped(wrap_pyfunction!(py_sample))?;
    m.add_wrapped(wrap_pyfunction!(py_repetition_penalty))?;
    
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greedy_python() {
        let logits = vec![1.0, 5.0, 3.0];
        assert_eq!(py_greedy(logits), 1);
    }
    
    #[test]
    fn test_top_k_python() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let result = py_top_k(logits, 2, 1.0, Some(42));
        assert!(result <= 3);
    }
}
