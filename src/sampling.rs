//! Fast LLM inference sampling primitives.

use rand::prelude::*;

/// Apply temperature scaling to logits (in-place).
#[inline]
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 0.0 {
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

/// Apply repetition penalty to logits.
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

/// Top-k indices: indices of k largest logits (unsorted within top-k).
#[inline]
pub fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(logits.len());
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap()
    });
    indices.truncate(k);
    indices
}

/// Top-k sampling: sample from the top-k tokens.
pub fn sample_top_k(logits: &mut [f32], k: usize, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    let k = k.min(logits.len());
    let indices = top_k_indices(logits, k);

    let max_logit = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = indices.iter().map(|&i| (logits[i] - max_logit).exp()).collect();
    let total: f32 = weights.iter().sum();
    if total == 0.0 {
        return indices[0] as u32;
    }
    for w in &mut weights {
        *w /= total;
    }

    let mut cumsum = 0.0;
    let r: f32 = rng.gen();
    for (pos, &token_id) in indices.iter().enumerate() {
        cumsum += weights[pos];
        if r < cumsum {
            return token_id as u32;
        }
    }
    *indices.last().unwrap() as u32
}

/// Nucleus (top-p) sampling: keep smallest token set with cumulative prob >= p.
pub fn sample_nucleus(logits: &mut [f32], p: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    if p <= 0.0 || p >= 1.0 {
        return sample_greedy(logits);
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    if sum == 0.0 {
        return sample_greedy(logits);
    }
    let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumsum = 0.0;
    let nucleus_end = indices.iter()
        .position(|&i| {
            cumsum += probs[i];
            cumsum >= p
        })
        .unwrap_or(indices.len());

    let nucleus = &indices[..nucleus_end.max(1)];
    let r = rng.gen::<f32>() * nucleus.len() as f32;
    *nucleus.get(r as usize).unwrap_or(&nucleus[nucleus.len() - 1]) as u32
}

/// Min-p sampling: keep tokens with probability > min_p * max_prob.
pub fn sample_min_p(logits: &mut [f32], min_p: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);
    if min_p <= 0.0 {
        return sample_greedy(logits);
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_logit + min_p.ln();

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
        return logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap_or(0);
    }

    let r = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for &(idx, p) in &candidates {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    candidates.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Typical sampling: sample from tokens whose log-probability is close to the entropy.
pub fn sample_typical(logits: &mut [f32], mass: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    apply_temperature(logits, temperature);

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    if sum == 0.0 {
        return sample_greedy(logits);
    }
    let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

    let entropy: f32 = probs.iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum();

    let target = entropy;

    let mut deviations: Vec<(usize, f32)> = (0..logits.len())
        .map(|i| {
            let log_p = if probs[i] > 0.0 { probs[i].ln() } else { f32::NEG_INFINITY };
            let dev = (log_p + target).abs();
            (i, dev)
        })
        .collect();

    deviations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n = ((1.0 - mass) * logits.len() as f32).ceil() as usize;
    let n = n.max(1).min(deviations.len());
    let candidates: Vec<usize> = deviations.into_iter().take(n).map(|(i, _)| i).collect();

    let cand_sum: f32 = candidates.iter().map(|&i| probs[i]).sum();
    if cand_sum == 0.0 {
        return candidates[0] as u32;
    }
    let r = rng.gen::<f32>() * cand_sum;
    let mut cumsum = 0.0;
    for &i in &candidates {
        cumsum += probs[i];
        if r < cumsum {
            return i as u32;
        }
    }
    *candidates.last().unwrap() as u32
}

/// Greedy decoding: pick the highest-probability token.
#[inline]
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32).unwrap_or(0)
}

/// Combined sampler.
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
    if let Some(penalty) = repetition_penalty {
        if let Some(tokens) = prev_tokens {
            apply_repetition_penalty(logits, tokens, penalty);
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));

    match method {
        "greedy" => sample_greedy(logits),
        "top_k" | "topk" => sample_top_k(logits, top_k.unwrap_or(50), temperature, &mut rng),
        "top_p" | "nucleus" => sample_nucleus(logits, top_p.unwrap_or(0.9), temperature, &mut rng),
        "min_p" => sample_min_p(logits, min_p.unwrap_or(0.1), temperature, &mut rng),
        "typical" => sample_typical(logits, top_p.unwrap_or(0.9), temperature, &mut rng),
        _ => sample_nucleus(logits, 0.9, temperature, &mut rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![0.0, 2.0, 1.0, -1.0];
        let result = sample_greedy(&logits);
        assert!(result == 1 || result == 2);
    }

    #[test]
    fn test_top_k_indices() {
        let logits = vec![1.0, 5.0, 3.0, -2.0, 4.0];
        let top3 = top_k_indices(&logits, 3);
        assert_eq!(top3.len(), 3);
        assert!(top3.contains(&1));
        assert!(top3.contains(&4));
        assert!(top3.contains(&2));
    }

    #[test]
    fn test_temperature() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[2] - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_repetition_penalty_positive() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let prev = vec![0u32, 1];
        apply_repetition_penalty(&mut logits, &prev, 2.0);
        assert!((logits[0] - 0.5).abs() < 0.001);
        assert!((logits[1] - 1.0).abs() < 0.001);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0);
    }

    #[test]
    fn test_repetition_penalty_negative() {
        let mut logits = vec![-2.0, -1.0, 3.0, 4.0];
        let prev = vec![0u32];
        apply_repetition_penalty(&mut logits, &prev, 2.0);
        assert!((logits[0] - (-4.0)).abs() < 0.001);
        assert_eq!(logits[1], -1.0);
    }

    #[test]
    fn test_top_k_sampling() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = sample_top_k(&mut logits, 2, 1.0, &mut rng);
        assert!(token <= 3);
    }

    #[test]
    fn test_nucleus_sampling() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = sample_nucleus(&mut logits, 0.9, 1.0, &mut rng);
        assert!(token <= 3);
    }

    #[test]
    fn test_min_p_sampling() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = sample_min_p(&mut logits, 0.1, 1.0, &mut rng);
        assert!(token <= 3);
    }
}
