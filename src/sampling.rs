//! Fast LLM inference sampling primitives.
//! All operations are in-place on logits where possible.

use rand::prelude::*;

/// Apply temperature scaling to logits (in-place).
/// Temperature of 0.0 = greedy (handled as special case).
/// Temperature of 1.0 = no change.
/// Lower = sharper distribution.
#[inline]
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 0.0 {
        // Greedy: set all but max to -inf
        if let Some(&max) = logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            for logit in logits.iter_mut() {
                *logit = if *logit == max { 0.0 } else { f32::NEG_INFINITY };
            }
        }
    } else if temperature != 1.0 {
        let scale = 1.0 / temperature;
        for logit in logits.iter_mut() {
            *logit *= scale;
        }
    }
}

/// Apply repetition penalty to logits, given previously generated token IDs.
/// Penalizes tokens that have already been generated.
#[inline]
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 || prev_tokens.is_empty() {
        return;
    }
    
    for (token_id, logit) in logits.iter_mut().enumerate() {
        if prev_tokens.contains(&(token_id as u32)) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Top-k sampling: keep only the k most probable tokens.
/// Returns the indices of kept tokens.
#[inline]
pub fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logits.len());
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    
    // Partial sort: get top k indices by logit value
    indices.select_nth_unstable_by(k, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap()
    });
    
    indices.truncate(k);
    indices
}

/// Top-k sampling: sample from the top-k tokens.
/// Returns the sampled token ID.
pub fn sample_top_k(logits: &mut [f32], k: usize, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    
    let k = k.min(logits.len());
    let indices = top_k_indices(logits, k);
    
    // Compute probabilities only over kept tokens
    let max_logit = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    
    let mut weights: Vec<f32> = indices.iter()
        .map(|&i| (logits[i] - max_logit).exp())
        .collect();
    
    let total: f32 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= total);
    
    // Sample from categorical distribution
    let mut cumsum = 0.0;
    let r: f32 = rng.gen();
    for &i in &indices {
        cumsum += weights[i.unwrap()];
        if r < cumsum {
            return i as u32;
        }
    }
    
    indices.last().copied().unwrap_or(0) as u32
}

/// Nucleus (top-p) sampling: keep tokens whose cumulative probability exceeds p.
/// Returns the sampled token ID.
pub fn sample_nucleus(logits: &mut [f32], p: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    
    if p <= 0.0 || p >= 1.0 {
        return sample_greedy(logits);
    }
    
    // Get max for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp and normalize
    let mut exp_logits: Vec<f32> = logits.iter()
        .map(|&l| (l - max_logit).exp())
        .collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter_mut().for_each(|w| *w /= sum);
    
    // Sort by probability descending
    let mut indices: Vec<usize> = (0..exp_logits.len()).collect();
    indices.sort_by(|&a, &b| exp_logits[b].partial_cmp(&exp_logits[a]).unwrap());
    
    // Find nucleus: smallest set of tokens with cumulative prob >= p
    let mut cumsum = 0.0;
    let mut nucleus_end = indices.len();
    for (i, &idx) in indices.iter().enumerate() {
        cumsum += exp_logits[idx];
        if cumsum >= p {
            nucleus_end = i + 1;
            break;
        }
    }
    
    // Sample uniformly from nucleus
    let nucleus = &indices[..nucleus_end];
    let r: f32 = rng.gen::<f32>() * nucleus.len() as f32;
    let sampled_idx = nucleus[r as usize].unwrap();
    
    sampled_idx as u32
}

/// Min-p sampling: keep tokens with probability > min_p * max_prob.
/// Returns the sampled token ID.
pub fn sample_min_p(logits: &mut [f32], min_p: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    
    if min_p <= 0.0 {
        return sample_greedy(logits);
    }
    
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_logit + (min_p.ln());
    
    // Compute exp and keep only tokens above threshold
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    let mut sum = 0.0f32;
    
    for (i, &logit) in logits.iter().enumerate() {
        if logit >= threshold {
            let p = (logit - max_logit).exp();
            candidates.push((i, p));
            sum += p;
        }
    }
    
    if candidates.is_empty() {
        return logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0);
    }
    
    // Normalize and sample
    let r: f32 = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (idx, p) in candidates {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    
    candidates.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Typical sampling: conditions on the entropy of the distribution.
/// Keeps tokens close to the expected entropy, then samples.
/// Returns the sampled token ID.
pub fn sample_typical(logits: &mut [f32], mass: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute log probabilities
    let log_probs: Vec<f32> = logits.iter()
        .map(|&l| {
            let shifted = l - max_logit;
            shifted - (shifted.exp() + max_logit.exp()).ln_approx()
        })
        .collect();
    
    // Entropy: -sum(p * log(p))
    let entropy: f32 = logits.iter()
        .zip(&log_probs)
        .map(|(&l, &lp)| {
            let p = (l - max_logit).exp();
            -p * lp
        })
        .sum();
    
    // Keep tokens whose log prob is close to -entropy
    let target = -entropy;
    let mut distances: Vec<(usize, f32)> = (0..logits.len())
        .map(|i| (i, (log_probs[i] - target).powi(2)))
        .collect();
    
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Take top mass
    let n = ((1.0 - mass) * logits.len() as f32).ceil() as usize;
    let n = n.max(1).min(distances.len());
    let candidates: Vec<usize> = distances.into_iter().take(n).map(|(i, _)| i).collect();
    
    // Sample uniformly
    let r = rng.gen::<f32>() * candidates.len() as f32;
    candidates[r as usize] as u32
}

/// Greedy decoding: always pick the highest-probability token.
#[inline]
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
}

/// Combined sampler that handles all methods in one call.
/// Returns the sampled token ID.
pub fn sample(
    logits: &mut [f32],
    method: &str,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    prev_tokens: Option<&[u32]>,
    seed: Option<u64>,
) -> u32 {
    // Apply repetition penalty first if specified
    if let Some(penalty) = repetition_penalty {
        if let Some(tokens) = prev_tokens {
            apply_repetition_penalty(logits, tokens, penalty);
        }
    }
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
    
    match method {
        "greedy" | "greedy" => sample_greedy(logits),
        "top_k" | "topk" => {
            let k = top_k.unwrap_or(50);
            sample_top_k(logits, k, temperature, &mut rng)
        }
        "top_p" | "nucleus" => {
            let p = top_p.unwrap_or(0.9);
            sample_nucleus(logits, p, temperature, &mut rng)
        }
        "min_p" => {
            let p = min_p.unwrap_or(0.1);
            sample_min_p(logits, p, temperature, &mut rng)
        }
        "typical" => {
            let mass = top_p.unwrap_or(0.9);
            sample_typical(logits, mass, temperature, &mut rng)
        }
        _ => sample_nucleus(logits, 0.9, temperature, &mut rng),
    }
}

/// Batch top-k: find top-k for multiple logit vectors simultaneously.
/// Useful for batched inference. Returns Vec of top-k index arrays.
pub fn batch_top_k<const K: usize>(logits_batch: &[[f32; K]]) -> Vec<[usize; K]> {
    todo!("SIMD-parallel top-k for fixed-size batches")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greedy() {
        let mut logits = vec![0.0, 2.0, 1.0, -1.0];
        assert_eq!(sample_greedy(&logits), 1);
        
        logits = vec![-5.0, 0.0, 0.0];
        assert_eq!(sample_greedy(&logits), 1); // first max
    }
    
    #[test]
    fn test_top_k_indices() {
        let logits = vec![1.0, 5.0, 3.0, -2.0, 4.0];
        let top3 = top_k_indices(&logits, 3);
        assert_eq!(top3.len(), 3);
        assert!(top3.contains(&1)); // 5.0
        assert!(top3.contains(&4)); // 4.0
        assert!(top3.contains(&2)); // 3.0
    }
    
    #[test]
    fn test_temperature() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[2] - 1.5).abs() < 0.001); // 3.0/2.0 = 1.5
    }
    
    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let prev = vec![2, 3];
        apply_repetition_penalty(&mut logits, &prev, 2.0);
        assert_eq!(logits[0], 1.0); // unchanged
        assert_eq!(logits[1], 1.0); // 2.0/2.0 = 1.0 (positive)
        assert_eq!(logits[3], 2.0); // 4.0/2.0 = 2.0 (positive)
    }
}
