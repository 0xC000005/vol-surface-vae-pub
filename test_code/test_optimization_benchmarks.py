"""
Optimization Performance Benchmarks

Measures actual speedup of vectorized implementations.
Run with --quick for fast check, --full for comprehensive benchmark.
"""

import torch
import time
import argparse
import numpy as np
from vae.full_covariance_prior import (
    build_ar1_covariance,
    build_ar1_cholesky_direct,
    kl_divergence_full_covariance
)


def benchmark_cholesky(device='cuda', num_iter=100, horizons=[30, 60, 90]):
    """
    Benchmark Cholesky construction across different horizons.

    This measures the current (vectorized) implementation speed.
    """
    print(f"\n{'='*60}")
    print(f"Cholesky Construction Benchmark (device={device})")
    print(f"{'='*60}")

    phi = torch.tensor(0.7, device=device)
    sigma = torch.tensor(1.0, device=device)

    results = {}

    for H in horizons:
        # Warmup
        for _ in range(10):
            build_ar1_cholesky_direct(phi, sigma, H, device)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iter):
            L = build_ar1_cholesky_direct(phi, sigma, H, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_per_call = elapsed / num_iter * 1000  # ms
        results[H] = time_per_call

        print(f"  H={H:3d}: {time_per_call:.4f} ms/call ({num_iter} iterations)")

    return results


def benchmark_kl(device='cuda', num_iter=50, batch_sizes=[16, 32], horizon=90, latent_dim=12):
    """
    Benchmark KL divergence computation.
    """
    print(f"\n{'='*60}")
    print(f"KL Divergence Benchmark (device={device}, H={horizon}, D={latent_dim})")
    print(f"{'='*60}")

    results = {}

    for B in batch_sizes:
        mu_q = torch.randn(B, horizon, latent_dim, device=device)
        logvar_q = torch.randn(B, horizon, latent_dim, device=device)
        mu_p = torch.randn(B, horizon, latent_dim, device=device)
        Sigma_p = build_ar1_covariance(0.7, 1.0, horizon, device)

        # Warmup
        for _ in range(5):
            kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iter):
            kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_per_call = elapsed / num_iter * 1000  # ms
        results[B] = time_per_call

        print(f"  B={B:3d}: {time_per_call:.4f} ms/call ({num_iter} iterations)")

    return results


def benchmark_gpu_memory():
    """
    Measure GPU memory usage for current implementation.
    """
    if not torch.cuda.is_available():
        print("\nGPU memory benchmark skipped (no CUDA)")
        return None

    print(f"\n{'='*60}")
    print(f"GPU Memory Usage")
    print(f"{'='*60}")

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create typical model tensors
    B, H, D = 32, 90, 12

    # Simulate forward pass data
    mu_q = torch.randn(B, H, D, device='cuda')
    logvar_q = torch.randn(B, H, D, device='cuda')
    mu_p = torch.randn(B, H, D, device='cuda')
    Sigma_p = build_ar1_covariance(0.7, 1.0, H, 'cuda')

    # Simulate KL computation
    kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)

    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    print(f"  Peak memory: {peak_memory:.2f} MB")

    return peak_memory


def main():
    parser = argparse.ArgumentParser(description='Benchmark optimizations')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (fewer iterations)')
    parser.add_argument('--full', action='store_true', help='Full benchmark suite')
    parser.add_argument('--cholesky-only', action='store_true', help='Only benchmark Cholesky')
    parser.add_argument('--kl-only', action='store_true', help='Only benchmark KL divergence')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("\nWARNING: Running on CPU - benchmarks will be slower")

    num_iter_chol = 20 if args.quick else 100
    num_iter_kl = 10 if args.quick else 50

    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.cholesky_only or not (args.kl_only or args.full):
        chol_results = benchmark_cholesky(device, num_iter_chol)

    if args.kl_only or not (args.cholesky_only or args.full):
        kl_results = benchmark_kl(device, num_iter_kl)

    if args.full:
        chol_results = benchmark_cholesky(device, num_iter_chol, horizons=[10, 30, 60, 90])
        kl_results = benchmark_kl(device, num_iter_kl, batch_sizes=[8, 16, 32, 64])
        mem_usage = benchmark_gpu_memory()

    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
