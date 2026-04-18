"""
Benchmark: inferlib (Rust) vs pure Python/NumPy sampling.
Run: python benchmarks/compare_vs_python.py
"""

import time
import numpy as np
import torch

try:
    import inferlib
    HAS_INFERLIB = True
except ImportError:
    HAS_INFERLIB = False
    print("WARNING: inferlib not installed. Run: pip install -e inferlib-python")


VOCAB_SIZE = 32000
SEQ_LEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50


def sample_top_k_numpy(logits: np.ndarray, k: int, temperature: float) -> int:
    """Pure NumPy top-k sampler."""
    if temperature != 1.0:
        logits = logits / temperature
    
    # Get top-k indices
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    
    # Softmax over top-k
    top_k_logits = top_k_logits - np.max(top_k_logits)
    probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    
    return np.random.choice(top_k_idx, p=probs)


def sample_top_p_numpy(logits: np.ndarray, p: float, temperature: float) -> int:
    """Pure NumPy nucleus (top-p) sampler."""
    if temperature != 1.0:
        logits = logits / temperature
    
    # Sort descending
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    
    # Softmax
    shifted = sorted_logits - np.max(sorted_logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    
    # Find nucleus
    cumsum = np.cumsum(probs)
    nucleus_end = np.searchsorted(cumsum, p) + 1
    nucleus_idx = sorted_idx[:nucleus_end]
    nucleus_probs = probs[:nucleus_end]
    
    # Renormalize
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    
    return np.random.choice(nucleus_idx, p=nucleus_probs)


def apply_repetition_penalty_numpy(logits: np.ndarray, prev_tokens: np.ndarray, penalty: float) -> np.ndarray:
    """Pure NumPy repetition penalty."""
    for token_id in prev_tokens:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def benchmark(fn, n_iters=1000, warmup=10):
    """Time a function in ms per call."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = time.perf_counter() - start
    
    return (elapsed / n_iters) * 1000  # ms


def main():
    np.random.seed(42)
    
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"iters: 1000 warmup: 10\n")
    
    logits_py = np.random.randn(VOCAB_SIZE).astype(np.float32)
    prev_tokens = np.random.randint(0, VOCAB_SIZE, size=SEQ_LEN)
    
    # --- NumPy baselines ---
    print("=== NumPy baselines ===")
    
    fn_numpy_topk = lambda: sample_top_k_numpy(logits_py.copy(), TOP_K, TEMPERATURE)
    t = benchmark(fn_numpy_topk)
    print(f"NumPy top-k:       {t:.4f} ms/call")
    
    fn_numpy_topp = lambda: sample_top_p_numpy(logits_py.copy(), TOP_P, TEMPERATURE)
    t = benchmark(fn_numpy_topp)
    print(f"NumPy top-p:       {t:.4f} ms/call")
    
    fn_numpy_rep = lambda: apply_repetition_penalty_numpy(logits_py.copy(), prev_tokens, 1.1)
    t = benchmark(fn_numpy_rep)
    print(f"NumPy rep-penalty: {t:.4f} ms/call")
    
    # --- Torch baselines ---
    print("\n=== PyTorch baselines ===")
    
    logits_th = torch.from_numpy(logits_py.copy())
    prev_th = torch.from_numpy(prev_tokens)
    
    def torch_topk():
        logits = logits_th.clone()
        if TEMPERATURE != 1.0:
            logits = logits / TEMPERATURE
        top_k_logits, top_k_idx = torch.topk(logits, TOP_K)
        probs = torch.softmax(top_k_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        return top_k_idx[idx].item()
    
    t = benchmark(torch_topk)
    print(f"torch top-k:       {t:.4f} ms/call")
    
    def torch_topp():
        logits = logits_th.clone()
        if TEMPERATURE != 1.0:
            logits = logits / TEMPERATURE
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        n = (cumsum < TOP_P).sum() + 1
        nucleus_probs = probs[:n]
        nucleus_idx = sorted_idx[:n]
        idx = torch.multinomial(nucleus_probs, 1)
        return nucleus_idx[idx].item()
    
    t = benchmark(torch_topp)
    print(f"torch top-p:       {t:.4f} ms/call")
    
    # --- inferlib (Rust) ---
    if HAS_INFERLIB:
        print("\n=== inferlib (Rust) ===")
        
        logits_list = logits_py.tolist()
        
        def rust_topk():
            return inferlib.top_k(logits_list.copy(), TOP_K, TEMPERATURE)
        
        t = benchmark(rust_topk)
        print(f"inferlib top-k:    {t:.4f} ms/call")
        
        def rust_topp():
            return inferlib.top_p(logits_list.copy(), TOP_P, TEMPERATURE)
        
        t = benchmark(rust_topp)
        print(f"inferlib top-p:    {t:.4f} ms/call")
        
        def rust_topp_with_rep():
            logits_copy = logits_list.copy()
            inferlib.repetition_penalty(logits_copy, prev_tokens.tolist(), 1.1)
            return inferlib.top_p(logits_copy, TOP_P, TEMPERATURE)
        
        t = benchmark(rust_topp_with_rep)
        print(f"inferlib rep+topp: {t:.4f} ms/call")
    else:
        print("\n=== inferlib (Rust) ===")
        print("Not available — run: cd inferlib-python && pip install -e .")


if __name__ == "__main__":
    main()
