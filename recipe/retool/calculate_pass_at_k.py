#!/usr/bin/env python3
"""
Calculate Pass@k values for evaluation results using the HuggingFace formula.
"""

import json
import itertools
import numpy as np
import argparse

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pass@k values from evaluation results")
    parser.add_argument("--results_path", type=str, required=True, 
                       help="Path to the evaluation results JSON file")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 32],
                       help="List of k values to calculate Pass@k for")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-problem statistics")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load the results
    with open(args.results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    n_samples = data['metadata']['n_samples']
    
    print(f"Loaded {len(results)} results with {n_samples} samples per problem")
    
    # Group results by problem_id
    problems = {}
    for result in results:
        pid = result['problem_id']
        if pid not in problems:
            problems[pid] = []
        problems[pid].append(result)
    
    # Sort samples within each problem by sample_id
    for pid in problems:
        problems[pid].sort(key=lambda x: x['sample_id'])
    
    print(f"Found {len(problems)} problems")
    
    # Calculate number of correct samples per problem
    num_correct_per_problem = []
    for pid, samples in problems.items():
        num_correct = sum(1 for s in samples if s['correct'])
        num_correct_per_problem.append(num_correct)
        if args.verbose:
            print(f"Problem {pid}: {num_correct}/{len(samples)} correct")
    
    if args.verbose:
        print(f"\nTotal correct per problem: {num_correct_per_problem}")
    print(f"Mean correct per problem: {np.mean(num_correct_per_problem):.2f}")
    
    # Calculate Pass@k for different k values
    print(f"\n=== Pass@k Calculation ===")
    for k in args.k_values:
        if k <= n_samples:
            pass_at_k_scores = estimate_pass_at_k(n_samples, num_correct_per_problem, k)
            pass_at_k = np.mean(pass_at_k_scores)
            print(f"Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
        else:
            print(f"Pass@{k}: N/A (k > n_samples)")
    
    # Also calculate simple accuracy
    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['correct'])
    accuracy = correct_samples / total_samples
    print(f"\nSimple accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()