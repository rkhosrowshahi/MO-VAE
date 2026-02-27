import os
import numpy as np
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pymoo.indicators.hv import HV

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: 'tabulate' not available. Using simple table formatting. Install with: pip install tabulate")

from utils.utils import get_dataset, set_seed
from models import get_network
from main import evaluate_with_recon_metrics, evaluate_generative_metrics


def load_model_from_checkpoint(model_path, dataset, arch, device):
    """
    Load model from checkpoint file.
    
    Args:
        model_path: Path to the checkpoint file (.pth)
        dataset: Dataset name (for getting input_size)
        arch: Architecture name (for verification)
        device: Device to load model on
        
    Returns:
        tuple: (model, args) where args is the configuration object
    """
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract args from checkpoint
    if 'args' in checkpoint:
        # Convert dict to Namespace-like object
        class Args:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        
        args = Args(checkpoint['args'])
    else:
        raise ValueError("Checkpoint does not contain 'args'. Cannot reconstruct model configuration.")
    
    # Verify architecture matches
    if hasattr(args, 'arch') and args.arch.lower() != arch.lower():
        print(f"Warning: Checkpoint arch ({args.arch}) does not match provided arch ({arch}). Using checkpoint arch.")
        arch = args.arch
    
    # Get dataset to determine input_size
    _, _, input_size = get_dataset(dataset, data_dir=args.data_dir if hasattr(args, 'data_dir') else './data', 
                                    normalize=getattr(args, 'normalize_inputs', getattr(args, 'normalize', False)))
    
    # Set arch in args if not present
    if not hasattr(args, 'arch'):
        args.arch = arch
    
    # Create model
    print(f"Creating {args.arch} model for {dataset} dataset (input_size={input_size})...")
    net = get_network(input_size, num_channels=3, args=args, device=device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    net.load_state_dict(model_state, strict=False)
    net.to(device)
    net.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Architecture: {args.arch}")
    print(f"  Total parameters: {net.total_trainable_params():,}")
    
    return net, args


def build_hv_indicator(objective_keys):
    """
    Build a Hypervolume (HV) indicator for multi-objective optimization evaluation.
    
    The hypervolume indicator measures the volume of the objective space dominated
    by a solution set. It's used to evaluate the quality of solutions in multi-objective
    optimization problems. Requires at least 2 objectives to compute.
    
    Uses a fixed reference point of 1.1 for all objectives.
    
    Args:
        objective_keys: Iterable of objective/loss names (e.g., ['reconstruction_loss', 'kl_loss'])
            
    Returns:
        HV: Hypervolume indicator object from pymoo, or None if fewer than 2 objectives
    """
    if len(objective_keys) < 2:
        return None

    num_objectives = len(objective_keys)
    
    # Always use fixed reference point of 1.1 for all objectives
    ref_point = [1.1] * num_objectives

    return HV(ref_point=np.array(ref_point))


def format_value(value, metric_name):
    """Format metric value for display."""
    if isinstance(value, float) and (value != value):  # Check for NaN
        return "N/A"
    
    if metric_name in ['ssim', 'precision', 'recall']:
        return f"{value:.4f}"
    elif metric_name in ['ssnr', 'psnr']:
        return f"{value:.2f} dB"
    elif metric_name in ['lpips', 'fid', 'rfid', 'gfid', 'kid', 'inception_score_mean', 'inception_score_std']:
        return f"{value:.4f}"
    elif metric_name == 'hv' or metric_name == 'hypervolume':
        return f"{value:.6e}"
    elif 'codebook_usage' in metric_name:
        return f"{value:.2f}%"
    else:
        # For loss values, use scientific notation for very small/large numbers
        if abs(value) < 1e-3 or abs(value) > 1e3:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"


def print_results_table(loss_meters, recon_metrics, gen_metrics, hv_value=None):
    """
    Print evaluation results in a formatted table.
    
    Args:
        loss_meters: Dictionary of AverageMeter objects from evaluate()
        recon_metrics: Dictionary from evaluate_recon_metrics() (rFID, PSNR, SSIM, LPIPS)
        gen_metrics: Dictionary from evaluate_generative_metrics() (gFID, IS, Precision, Recall, KID)
        hv_value: Hypervolume value (optional)
    """
    # Add loss metrics (training objectives)
    print("\n" + "="*80)
    print("TEST LOSSES (Training Objectives)")
    print("="*80)
    
    loss_data = []
    for key, meter in loss_meters.items():
        value = meter.avg if hasattr(meter, 'avg') else meter
        loss_data.append([key, format_value(value, key)])
    
    # Add HV if available
    if hv_value is not None:
        loss_data.append(["Hypervolume (HV)", format_value(hv_value, 'hv')])
    
    # Print loss table
    if loss_data:
        if HAS_TABULATE:
            print(tabulate(loss_data, headers=["Metric", "Value"], tablefmt="grid"))
        else:
            max_name_len = max(len(name) for name, _ in loss_data)
            print(f"{'Metric':<{max_name_len+5}} {'Value':>20}")
            print("-" * (max_name_len + 26))
            for name, value in loss_data:
                print(f"{name:<{max_name_len+5}} {value:>20}")
    
    # Reconstruction metrics (rFID, PSNR, SSIM, LPIPS)
    print("\n" + "="*80)
    print("RECONSTRUCTION METRICS")
    print("="*80)
    recon_data = [
        ["rFID", format_value(recon_metrics.get('rfid', float('nan')), 'rfid')],
        ["PSNR", format_value(recon_metrics.get('psnr', float('nan')), 'psnr')],
        ["SSIM", format_value(recon_metrics.get('ssim', float('nan')), 'ssim')],
        ["LPIPS", format_value(recon_metrics.get('lpips', float('nan')), 'lpips')],
    ]
    if HAS_TABULATE:
        print(tabulate(recon_data, headers=["Metric", "Value"], tablefmt="grid"))
    else:
        max_name_len = max(len(name) for name, _ in recon_data)
        print(f"{'Metric':<{max_name_len+5}} {'Value':>20}")
        print("-" * (max_name_len + 26))
        for name, value in recon_data:
            print(f"{name:<{max_name_len+5}} {value:>20}")
    
    # Generative metrics (gFID, IS, Precision, Recall, KID)
    print("\n" + "="*80)
    print("GENERATIVE METRICS")
    print("="*80)
    gen_data = [
        ["gFID", format_value(gen_metrics.get('gfid', float('nan')), 'gfid')],
        ["IS Mean", format_value(gen_metrics.get('inception_score_mean', float('nan')), 'inception_score_mean')],
        ["IS Std", format_value(gen_metrics.get('inception_score_std', float('nan')), 'inception_score_std')],
        # ["Precision", format_value(gen_metrics.get('precision', float('nan')), 'precision')],  # Commented out
        # ["Recall", format_value(gen_metrics.get('recall', float('nan')), 'recall')],              # Commented out
        ["KID", format_value(gen_metrics.get('kid', float('nan')), 'kid')],
    ]
    if HAS_TABULATE:
        print(tabulate(gen_data, headers=["Metric", "Value"], tablefmt="grid"))
    else:
        max_name_len = max(len(name) for name, _ in gen_data)
        print(f"{'Metric':<{max_name_len+5}} {'Value':>20}")
        print("-" * (max_name_len + 26))
        for name, value in gen_data:
            print(f"{name:<{max_name_len+5}} {value:>20}")
    
    print("="*80 + "\n")


def evaluate(arch, dataset, model_path, device=None, batch_size=128, num_workers=0, 
             max_fid_samples=5000, max_gen_metrics_samples=5000, seed=None, verbose=True):
    """
    Evaluate a trained model on test set.
    
    Args:
        arch: Model architecture (e.g., 'vae', 'vq_vae', 'gg_vae', 'betatc_vae')
        dataset: Dataset name (e.g., 'CIFAR10', 'CIFAR100', 'CelebA', 'ImageNet')
        model_path: Path to the model checkpoint file (.pth)
        device: Device to run evaluation on (default: 'cuda:0' if available, else 'cpu')
        batch_size: Batch size for evaluation (default: 128)
        num_workers: Number of data loading workers (default: 0)
        max_fid_samples: Maximum number of samples for FID computation (default: 5000)
        max_gen_metrics_samples: Maximum number of samples for generative metrics computation (default: 5000)
        seed: Random seed for reproducibility (default: None)
        verbose: Whether to print progress and results (default: True)
        
    Returns:
        dict: Dictionary containing all evaluation results:
            - 'test_losses': Dictionary of test loss values (training objectives)
            - 'hv': Hypervolume value (or None if < 2 objectives)
            - 'recon_metrics': rFID, PSNR, SSIM, LPIPS
            - 'generative_metrics': gFID, IS, Precision, Recall, KID
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
    
    device = torch.device(device)
    if verbose:
        print(f"Using device: {device}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model from checkpoint
    net, checkpoint_args = load_model_from_checkpoint(
        model_path, 
        dataset, 
        arch, 
        device
    )
    
    # Get dataset
    if verbose:
        print(f"\nLoading {dataset} dataset...")
    train_dataset, test_dataset, input_size = get_dataset(
        dataset, 
        data_dir=getattr(checkpoint_args, 'data_dir', './data'),
        normalize=getattr(checkpoint_args, 'normalize_inputs', getattr(checkpoint_args, 'normalize', False))
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    if verbose:
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Create evaluation args object
    class EvalArgs:
        def __init__(self):
            self.max_fid_samples = max_fid_samples
            self.max_gen_metrics_samples = max_gen_metrics_samples
            self.batch_size = batch_size
    
    eval_args = EvalArgs()
    
    # Single pass: test losses + reconstruction metrics (rFID, PSNR, SSIM, LPIPS)
    if verbose:
        print("\n" + "="*80)
        print("Evaluating test losses and reconstruction metrics (single pass)...")
        print("="*80)
    loss_meters, recon_metrics = evaluate_with_recon_metrics(net, test_loader, device=device, args=eval_args)
    
    # Convert loss_meters to dictionary of values (losses and codebook_usage only)
    test_losses = {}
    for key, meter in loss_meters.items():
        test_losses[key] = meter.avg if hasattr(meter, 'avg') else meter
    
    # Calculate Hypervolume (HV) for test losses
    objective_keys = list(net.objectives.keys())
    hv_indicator = build_hv_indicator(objective_keys)
    hv_value = None
    if hv_indicator is not None:
        eval_point = np.array([[loss_meters[key].avg for key in objective_keys]])
        hv_value = hv_indicator(eval_point)
        if verbose:
            print(f"Hypervolume (HV): {hv_value:.6e}")
    else:
        if verbose:
            print(f"Hypervolume (HV): N/A (requires at least 2 objectives, found {len(objective_keys)})")
    
    # Evaluate generative metrics (gFID, IS, Precision, Recall, KID)
    if verbose:
        print("\n" + "="*80)
        print("Evaluating generative metrics...")
        print("="*80)
    gen_metrics = evaluate_generative_metrics(net, test_loader, device=device, args=eval_args)
    
    # Print results in table format if verbose
    if verbose:
        print_results_table(loss_meters, recon_metrics, gen_metrics, hv_value=hv_value)
        print("Evaluation completed!")
    
    # Return results as dictionary
    results = {
        'test_losses': test_losses,
        'hv': hv_value,
        'recon_metrics': recon_metrics,
        'generative_metrics': gen_metrics,
        'arch': arch,
        'dataset': dataset,
        'model_path': model_path,
    }
    
    return results


def main():
    """Command-line interface for evaluate function."""
    parser = ArgumentParser(description="Evaluate a trained model on test set")
    
    parser.add_argument("--arch", type=str, required=True,
                        help="Model architecture (e.g., 'vae', 'vq_vae', 'gg_vae', 'betatc_vae')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., 'CIFAR10', 'CIFAR100', 'CelebA', 'ImageNet')")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint file (.pth)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    parser.add_argument("--max_fid_samples", type=int, default=5000,
                        help="Maximum number of samples for FID computation")
    parser.add_argument("--max_gen_metrics_samples", type=int, default=5000,
                        help="Maximum number of samples for generative metrics computation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Call evaluate function with parsed arguments
    results = evaluate(
        arch=args.arch,
        dataset=args.dataset,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_fid_samples=args.max_fid_samples,
        max_gen_metrics_samples=args.max_gen_metrics_samples,
        seed=args.seed,
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    main()

