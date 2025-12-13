"""
Benchmark script to test different numbers of DataLoader workers for CelebA-HQ dataset.

This script loads the CelebA-HQ dataset using the get_dataset function from utils/utils.py
and tests various worker counts to determine the optimal number for data loading speed.
"""

import time
import torch
from torch.utils.data import DataLoader
from utils.utils import get_dataset


def benchmark_workers(
    dataset_name='CelebA-HQ',
    data_dir='./data',
    normalize=False,
    batch_size=16,
    num_batches=100,
    worker_counts=[0, 1, 2, 4, 6, 8, 12, 16],
    pin_memory=True,
    shuffle=True,
    num_warmup_batches=5,
    num_runs=3,
    device=None
):
    """
    Benchmark different numbers of DataLoader workers.
    
    Args:
        dataset_name: Name of the dataset (default: 'CelebA-HQ')
        data_dir: Directory containing the dataset
        normalize: Whether to normalize the dataset
        batch_size: Batch size to use for testing
        num_batches: Number of batches to iterate through for timing
        worker_counts: List of worker counts to test
        pin_memory: Whether to use pin_memory in DataLoader
        shuffle: Whether to shuffle the dataset
        num_warmup_batches: Number of batches to skip before timing (warmup)
        num_runs: Number of times to run each test for averaging
        device: PyTorch device to transfer batches to (default: cuda:0 if available, else cpu)
    
    Returns:
        dict: Dictionary mapping worker counts to average time per batch
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Loading {dataset_name} dataset...")
    train_dataset, test_dataset, input_size = get_dataset(
        dataset_name, 
        data_dir=data_dir, 
        normalize=normalize
    )
    
    print(f"Dataset loaded: {len(train_dataset)} training samples")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Number of batches to test: {num_batches}")
    print(f"Warmup batches: {num_warmup_batches}")
    print(f"Number of runs per worker count: {num_runs}")
    print("-" * 60)
    
    results = {}
    
    for num_workers in worker_counts:
        print(f"\nTesting {num_workers} worker(s)...")
        
        # Create DataLoader with current worker count
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory if device.type == 'cuda' else False,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        run_times = []
        
        for run in range(num_runs):
            # Warmup: iterate through a few batches to initialize workers
            loader_iter = iter(train_loader)
            for _ in range(num_warmup_batches):
                try:
                    images, labels = next(loader_iter)
                    # Transfer to device during warmup too
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    try:
                        images, labels = next(loader_iter)
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    except StopIteration:
                        # Dataset is empty or exhausted, cannot continue warmup
                        print(f"  Warning: Dataset exhausted during warmup. Skipping remaining warmup batches.")
                        break
            
            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Actual timing
            start_time = time.time()
            batch_count = 0
            
            loader_iter = iter(train_loader)
            while batch_count < num_batches:
                try:
                    images, labels = next(loader_iter)
                    # Transfer to GPU (this is what pin_memory helps with)
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    batch_count += 1
                except StopIteration:
                    loader_iter = iter(train_loader)
                    try:
                        images, labels = next(loader_iter)
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        batch_count += 1
                    except StopIteration:
                        # Dataset is empty or exhausted, cannot continue timing
                        print(f"  Warning: Dataset exhausted after {batch_count} batches. "
                              f"Requested {num_batches} batches but dataset only provided {batch_count}.")
                        break
            
            # Synchronize GPU before ending timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            time_per_batch = elapsed_time / num_batches
            run_times.append(time_per_batch)
            
            print(f"  Run {run + 1}/{num_runs}: {time_per_batch:.4f}s per batch "
                  f"({elapsed_time:.2f}s total)")
        
        avg_time = sum(run_times) / len(run_times)
        std_time = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5
        results[num_workers] = {
            'avg_time_per_batch': avg_time,
            'std_time_per_batch': std_time,
            'all_times': run_times
        }
        
        print(f"  Average: {avg_time:.4f}s ± {std_time:.4f}s per batch")
    
    return results


def print_results(results):
    """Print benchmarking results in a formatted table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Workers':<10} {'Avg Time/Batch (s)':<20} {'Std Dev (s)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    # Sort by worker count
    sorted_results = sorted(results.items())
    
    # Find the baseline (usually 0 workers or the fastest)
    baseline_time = min(r['avg_time_per_batch'] for r in results.values())
    
    for num_workers, stats in sorted_results:
        avg_time = stats['avg_time_per_batch']
        std_time = stats['std_time_per_batch']
        speedup = baseline_time / avg_time
        
        print(f"{num_workers:<10} {avg_time:<20.4f} {std_time:<15.4f} {speedup:<10.2f}x")
    
    # Find and highlight the fastest
    fastest_workers = min(results.items(), key=lambda x: x[1]['avg_time_per_batch'])
    print("-" * 70)
    print(f"\nFastest configuration: {fastest_workers[0]} worker(s) "
          f"({fastest_workers[1]['avg_time_per_batch']:.4f}s per batch)")
    print("=" * 70)


def main():
    """Main function to run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark DataLoader workers for CelebA-HQ dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CelebA-HQ',
        help='Dataset name (default: CelebA-HQ). Options: CIFAR10, CIFAR100, CelebA, CelebA-HQ, ImageNet'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data',
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '--normalize', 
        action='store_true',
        help='Normalize the dataset'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16,
        help='Batch size for testing (default: 16)'
    )
    parser.add_argument(
        '--num_batches', 
        type=int, 
        default=100,
        help='Number of batches to iterate through (default: 100)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        nargs='+',
        default=[0, 1, 2, 4, 6, 8, 12, 16],
        help='Worker counts to test (default: 0 1 2 4 6 8 12 16)'
    )
    parser.add_argument(
        '--no_pin_memory', 
        action='store_true',
        help='Disable pin_memory (useful for CPU-only testing)'
    )
    parser.add_argument(
        '--no_shuffle', 
        action='store_true',
        help='Disable shuffling'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        default=5,
        help='Number of warmup batches (default: 5)'
    )
    parser.add_argument(
        '--runs', 
        type=int, 
        default=3,
        help='Number of runs per worker count (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda:0 if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
        if not args.no_pin_memory:
            print("Note: pin_memory will be disabled for CPU.")
    
    print(f"\nStarting benchmark with the following settings:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Normalize: {args.normalize}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Worker counts to test: {args.workers}")
    print(f"  Pin memory: {not args.no_pin_memory}")
    print(f"  Shuffle: {not args.no_shuffle}")
    print()
    
    # Determine device
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    results = benchmark_workers(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        normalize=args.normalize,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        worker_counts=args.workers,
        pin_memory=not args.no_pin_memory,
        shuffle=not args.no_shuffle,
        num_warmup_batches=args.warmup,
        num_runs=args.runs,
        device=device
    )
    
    print_results(results)


if __name__ == '__main__':
    main()

