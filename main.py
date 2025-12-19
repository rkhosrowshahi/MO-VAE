import os
import time
from argparse import ArgumentParser

# Load .env file if it exists (for WANDB_API_KEY), otherwise use existing wandb login
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchjd.autojac import backward, mtl_backward
from torchjd.aggregation import (
    UPGrad,
    PCGrad,
    Mean,
    AlignedMTL,
    NashMTL,
    IMTLG,
    MGDA,
    CAGrad,
    DualProj,
    Sum
)

import wandb
from pymoo.indicators.hv import HV
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import scienceplots
from torchvision.utils import make_grid

from utils.utils import AverageMeter, get_dataset, set_seed
from utils.metrics import ssim, ssnr, psnr, lpips, calculate_fid, calculate_inception_score, calculate_precision_recall
from models import get_network

plt.style.use(["science", "ieee", "no-latex"])

# Global step tracker for wandb logging in hooks
_current_step = 0


def print_weights(_, __, weights: torch.Tensor) -> None:
    """
    Hook function to log task weights from multi-task learning aggregator.
    
    This function is registered as a forward hook on the aggregator's weighting
    module to track how weights are assigned to different tasks during training.
    The weights are logged to wandb for visualization.
    
    Args:
        _: Module instance (unused, required by hook signature)
        __: Input to the module (unused, required by hook signature)
        weights: Tensor of shape (num_tasks,) containing the weight for each task
        
    Returns:
        None
    """
    global _current_step
    # print(f"Weights: {weights}")
    if wandb.run is not None:
        log_dict = {f"train/task_{i}_weight": weight.item() for i, weight in enumerate(weights)}
        wandb.log(log_dict, step=_current_step)


def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], weights: torch.Tensor) -> None:
    """
    Hook function to compute and log cosine similarity between weighted and mean gradients.
    
    This function is registered as a forward hook to monitor how similar the weighted
    aggregated gradient (from multi-task learning) is to the simple average gradient.
    Higher similarity indicates the aggregator is producing gradients close to uniform
    averaging, while lower similarity suggests more task-specific weighting.
    
    Args:
        _: Module instance (unused, required by hook signature)
        inputs: Tuple containing gradient matrix of shape (num_tasks, num_params)
        weights: Tensor of shape (num_tasks,) containing task weights
        
    Returns:
        None
    """
    global _current_step
    matrix = inputs[0]
    # Compute mean gradient (simple average across tasks)
    gd_output = matrix.mean(dim=0)
    # Compute weighted aggregated gradient
    # matrix shape: [num_tasks, num_params], weights shape: [num_tasks]
    weighted_grad = (matrix.T @ weights)  # Shape: [num_params]
    similarity = torch.nn.functional.cosine_similarity(weighted_grad, gd_output, dim=0)
    similarity_value = similarity.item()
    # print(f"Cosine similarity: {similarity_value:.4f}")
    if wandb.run is not None:
        wandb.log({"train/gradient_similarity": similarity_value}, step=_current_step)


def train_epoch(net, train_loader, optimizer, aggregator, step, device, args):
    """
    Train the model for one epoch.
    
    Performs a full training pass over the dataset, computing losses, applying
    multi-task learning aggregation if specified, and updating model parameters.
    Tracks all loss components and logs them to wandb if enabled.
    
    Args:
        net: The neural network model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        aggregator: Multi-task learning aggregator (None, "sum", or aggregator object)
        step: Current global training step (will be updated)
        device: Device to run training on (e.g., 'cuda:0' or 'cpu')
        args: Configuration object containing training hyperparameters
        
    Returns:
        tuple: (loss_meters, updated_step)
            - loss_meters: Dictionary mapping loss names to AverageMeter objects
            - updated_step: Updated global step counter after this epoch
    """
    net.train()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()
    # Track codebook usage percentage for VQ-VAE models
    codebook_usage_meter = AverageMeter()

    for images, _ in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss_dict = net.loss_function(images, args=outputs)
        total_loss = sum(loss_dict.values())

        if total_loss.item() > 1e15:
            tqdm.write(f"Step {step}: EXPLODING: Total loss: {total_loss.item():.6e}, Losses: {loss_dict}")

        # Extract codebook_usage_percentage if available (for VQ-VAE models)
        if "codebook_usage_percentage" in outputs:
            codebook_usage_meter.update(outputs["codebook_usage_percentage"], n=images.size(0))

        # Update global step for hooks before backward
        global _current_step
        _current_step = step + 1
        
        if aggregator is None or aggregator == "sum":
            total_loss.backward()
        else:
            features = None
            if net.features is not None:
                features = [outputs[feature] for feature in net.features]
            
            if features is not None:
                mtl_backward(
                    losses=list(loss_dict.values()),
                    features=features,
                    aggregator=aggregator,
                    retain_graph=True,
                )
            else:
                backward(loss_dict.values(), aggregator=aggregator)

        # Clip gradients to prevent numerical instabilities
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_grad_norm)

        optimizer.step()

        loss_meters["total_loss"].update(total_loss.item())
        for key, value in loss_dict.items():
            loss_meters[key].update(value.item())

        step += 1
        if args.use_wandb:
            log_dict = {
                **{f"train/{key}": meter.avg for key, meter in loss_meters.items()},
                **{f"train/{key}_curr": meter.val for key, meter in loss_meters.items()},
            }
            # Add codebook usage percentage if available
            if codebook_usage_meter.count > 0:
                log_dict["train/codebook_usage_percentage"] = codebook_usage_meter.avg
            wandb.log(log_dict, step=step)

    # Return codebook usage meter if it was tracked
    if codebook_usage_meter.count > 0:
        loss_meters["codebook_usage_percentage"] = codebook_usage_meter
    
    return loss_meters, step


def evaluate(net, data_loader, device, args):
    """
    Evaluate the model on a dataset.
    
    Computes loss metrics and image quality metrics (SSIM, SSNR, PSNR, LPIPS, FID) on the
    provided dataset. The model is set to evaluation mode and gradients are
    disabled for efficiency.
    
    Args:
        net: The neural network model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on (e.g., 'cuda:0' or 'cpu')
        args: Configuration object containing evaluation settings
            - max_fid_samples: Maximum number of samples to use for FID computation
            
    Returns:
        dict: Dictionary mapping metric names to AverageMeter objects containing:
            - All loss components (e.g., 'reconstruction_loss', 'kl_loss', 'total_loss')
            - 'ssim': Structural Similarity Index (higher is better, range 0-1)
            - 'ssnr': Signal-to-Noise Ratio in dB (higher is better)
            - 'psnr': Peak Signal-to-Noise Ratio in dB (higher is better)
            - 'lpips': Learned Perceptual Image Patch Similarity (lower is better, lower bound is 0)
            - 'fid': Fréchet Inception Distance (lower is better, lower bound is 0)
    """
    net.eval()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()
    
    # Metrics meters
    ssim_meter = AverageMeter()
    ssnr_meter = AverageMeter()
    psnr_meter = AverageMeter()
    lpips_meter = AverageMeter()
    codebook_usage_meter = AverageMeter()
    
    # Collect images for FID computation
    all_real_images = []
    all_recon_images = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = net(images)
            loss_dict = net.loss_function(images, args=outputs)
            total_loss = sum(loss_dict.values())

            loss_meters["total_loss"].update(total_loss.item())
            for key, value in loss_dict.items():
                loss_meters[key].update(value.item())
            
            # Extract codebook_usage_percentage if available (for VQ-VAE models)
            if "codebook_usage_percentage" in outputs:
                codebook_usage_meter.update(outputs["codebook_usage_percentage"], n=images.size(0))
            
            # Compute SSIM, SSNR, PSNR, and LPIPS
            recons = outputs.get("recons")
            if recons is not None:
                # Compute SSIM per batch
                batch_ssim = ssim(images, recons, size_average=True)
                ssim_meter.update(batch_ssim.item(), n=images.size(0))
                
                # Compute SSNR per batch
                batch_ssnr = ssnr(images, recons)
                ssnr_meter.update(batch_ssnr, n=images.size(0))
                
                # Compute PSNR per batch
                batch_psnr = psnr(images, recons)
                psnr_meter.update(batch_psnr, n=images.size(0))
                
                # Compute LPIPS per batch
                batch_lpips = lpips(images, recons, device=device)
                lpips_meter.update(batch_lpips, n=images.size(0))
                
                # Collect images for FID (limit to avoid memory issues)
                max_fid_samples = args.max_fid_samples
                current_samples = sum(img.size(0) for img in all_real_images)
                if current_samples + images.size(0) <= max_fid_samples:
                    all_real_images.append(images.cpu())
                    all_recon_images.append(recons.cpu())

    # Compute FID if we have collected images
    # Note: FID is a distance/error metric (lower is better), unlike SSIM/SSNR (higher is better)
    fid_distance = float('nan')
    if len(all_real_images) > 0:
        try:
            all_real = torch.cat(all_real_images, dim=0)
            all_recon = torch.cat(all_recon_images, dim=0)
            # Limit to same number of samples for fair comparison
            min_samples = min(len(all_real), len(all_recon), max_fid_samples)
            all_real = all_real[:min_samples]
            all_recon = all_recon[:min_samples]
            fid_distance = calculate_fid(all_real, all_recon, device=device, batch_size=50)
        except Exception as e:
            tqdm.write(f"Warning: FID computation failed: {e}")
            fid_distance = float('nan')
    
    # Add metrics to loss_meters for consistent return format
    # SSIM: similarity score (higher is better, range 0-1)
    # SSNR: signal-to-noise ratio in dB (higher is better)
    # PSNR: peak signal-to-noise ratio in dB (higher is better)
    # LPIPS: Learned Perceptual Image Patch Similarity (lower is better, lower bound is 0)
    # FID: Fréchet Inception Distance (lower is better, lower bound is 0)
    loss_meters["ssim"] = ssim_meter
    loss_meters["ssnr"] = ssnr_meter
    loss_meters["psnr"] = psnr_meter
    loss_meters["lpips"] = lpips_meter
    loss_meters["fid"] = AverageMeter()
    loss_meters["fid"].update(fid_distance)
    # Add codebook usage percentage if available
    if codebook_usage_meter.count > 0:
        loss_meters["codebook_usage_percentage"] = codebook_usage_meter

    return loss_meters


def generate_random_samples(net, num_samples, device, save_path=None, log_to_wandb=False, epoch=None, step=None):
    """
    Generate random samples from the model's latent space.
    
    Samples random latent vectors and decodes them to generate new images.
    Optionally saves the samples as a grid image and logs to wandb.
    
    Args:
        net: The neural network model (must have a sample() method)
        num_samples: Number of random samples to generate
        device: Device to run generation on (e.g., 'cuda:0' or 'cpu')
        save_path: Optional path to save the generated samples grid image
        log_to_wandb: Whether to log the samples to wandb
        epoch: Current epoch number (for logging)
        step: Current global step (for logging)
        
    Returns:
        torch.Tensor: Generated samples tensor of shape (num_samples, C, H, W)
    """
    samples = net.sample(num_samples=num_samples, device=device)
    grid = make_grid(samples, nrow=int(np.sqrt(num_samples)), normalize=True)

    # Convert tensor to numpy array (for both matplotlib and wandb, avoids Windows temp directory issues)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    grid_np = np.clip(grid_np, 0, 1)

    if save_path is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title("Generated Random Samples")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    if log_to_wandb and epoch is not None:
        wandb.log({"generated_samples": wandb.Image(grid_np, caption=f"Epoch {epoch}")}, step=step)

    return samples


def generate_reconstructed_samples(
    net,
    split_name,
    loader,
    num_samples,
    device,
    save_path=None,
    log_to_wandb=False,
    epoch=None,
    step=None,
):
    """
    Generate reconstruction samples from a dataset.
    
    Takes images from the provided data loader, passes them through the model,
    and creates a side-by-side comparison grid of originals and reconstructions.
    Optionally saves the grid and logs to wandb.
    
    Args:
        net: The neural network model to use for reconstruction
        split_name: Name of the dataset split (e.g., 'train', 'test') for labeling
        loader: DataLoader containing images to reconstruct
        num_samples: Number of samples to reconstruct and display
        device: Device to run reconstruction on (e.g., 'cuda:0' or 'cpu')
        save_path: Optional path to save the comparison grid image
        log_to_wandb: Whether to log the samples to wandb
        epoch: Current epoch number (for logging)
        step: Current global step (for logging)
        
    Returns:
        tuple: (originals, reconstructions)
            - originals: Tensor of original images (num_samples, C, H, W)
            - reconstructions: Tensor of reconstructed images (num_samples, C, H, W)
            
    Raises:
        KeyError: If model output dictionary doesn't contain 'recons' key
    """
    net.eval()
    originals, reconstructions = [], []

    with torch.no_grad():
        collected = 0
        for images, _ in loader:
            if collected >= num_samples:
                break

            images = images.to(device)
            take = min(images.size(0), num_samples - collected)
            batch = images[:take]

            outputs = net(batch)
            recon = outputs.get("recons")
            if recon is None:
                raise KeyError("Model output dictionary must contain 'recons'.")

            originals.append(batch.cpu())
            reconstructions.append(recon.cpu())
            collected += take

    originals = torch.cat(originals, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0)

    comparison = []
    for idx in range(num_samples):
        comparison.append(originals[idx])
        comparison.append(reconstructions[idx])
    comparison_tensor = torch.stack(comparison)
    grid = make_grid(comparison_tensor, nrow=int(np.sqrt(num_samples)) * 2, normalize=True)

    # Convert tensor to numpy array (for both matplotlib and wandb, avoids Windows temp directory issues)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    grid_np = np.clip(grid_np, 0, 1)

    if save_path is not None:
        plt.figure(figsize=(15, 15))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title(f"Reconstructed {split_name} Samples (Original | Reconstructed)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    if log_to_wandb and epoch is not None:
        wandb.log(
            {f"reconstructed_{split_name}_samples": wandb.Image(grid_np, caption=f"Epoch {epoch}")},
            step=step,
        )

    return originals, reconstructions


def build_hv_indicator(objective_keys, args):
    """
    Build a Hypervolume (HV) indicator for multi-objective optimization evaluation.
    
    The hypervolume indicator measures the volume of the objective space dominated
    by a solution set. It's used to evaluate the quality of solutions in multi-objective
    optimization problems. Requires at least 2 objectives to compute.
    
    Args:
        objective_keys: Iterable of objective/loss names (e.g., ['reconstruction_loss', 'kl_loss'])
        args: Configuration object that may contain:
            - hv_ref: List of reference point values for each objective (optional)
                     If not provided, defaults to 1.1 for all objectives
            
    Returns:
        HV: Hypervolume indicator object from pymoo, or None if fewer than 2 objectives
    """
    if len(objective_keys) < 2:
        return None

    num_objectives = len(objective_keys)
    
    # Use hv_ref list if provided and has correct length, otherwise default to 1.1 for all objectives
    if hasattr(args, 'hv_ref') and args.hv_ref is not None and len(args.hv_ref) == num_objectives:
        ref_point = args.hv_ref
    else:
        ref_point = [1.1] * num_objectives

    return HV(ref_point=np.array(ref_point))


def evaluate_generative_metrics(net, test_loader, device, args):
    """
    Evaluate all generative model metrics on generated samples.
    
    Generates samples from the model and computes comprehensive evaluation metrics:
    - SSIM, SSNR, PSNR, LPIPS: Perceptual and pixel-level quality metrics (comparing generated vs real)
    - FID: Fréchet Inception Distance between generated and real image distributions
    - IS: Inception Score measuring quality and diversity of generated images
    - Precision: Fraction of generated images that are realistic
    - Recall: Fraction of real images covered by the generated distribution
    
    Args:
        net: The neural network model (must have a sample() method)
        test_loader: DataLoader for test/real images
        device: Device to run evaluation on (e.g., 'cuda:0' or 'cpu')
        args: Configuration object containing:
            - max_gen_metrics_samples: Maximum number of samples to generate/use
            
    Returns:
        dict: Dictionary containing all computed metrics:
            - 'ssim': Structural Similarity Index (higher is better, range 0-1)
            - 'ssnr': Signal-to-Noise Ratio in dB (higher is better)
            - 'psnr': Peak Signal-to-Noise Ratio in dB (higher is better)
            - 'lpips': Learned Perceptual Image Patch Similarity (lower is better)
            - 'fid': Fréchet Inception Distance (lower is better)
            - 'inception_score_mean': Mean IS across splits (higher is better)
            - 'inception_score_std': Standard deviation of IS
            - 'precision': Precision score (0-1, higher is better)
            - 'recall': Recall score (0-1, higher is better)
    """
    tqdm.write("Evaluating all generative metrics (SSIM, SSNR, PSNR, LPIPS, FID, IS, Precision, Recall)...")
    
    num_samples = args.max_gen_metrics_samples
    
    # Generate samples from the model
    tqdm.write(f"Generating {num_samples} samples from the model...")
    generated_samples = []
    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating samples", leave=False):
            batch_size_actual = min(batch_size, num_samples - len(generated_samples))
            if batch_size_actual <= 0:
                break
            samples = net.sample(num_samples=batch_size_actual, device=device)
            generated_samples.append(samples.cpu())
    
    generated_images = torch.cat(generated_samples, dim=0)[:num_samples]
    
    # Collect real images from test dataset
    tqdm.write(f"Collecting {num_samples} real images from test dataset...")
    real_images = []
    collected = 0
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Collecting real images", leave=False):
            if collected >= num_samples:
                break
            take = min(images.size(0), num_samples - collected)
            real_images.append(images[:take].cpu())
            collected += take
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Ensure both are in [0, 1] range (metrics functions handle normalization, but let's be safe)
    generated_images = torch.clamp(generated_images, 0, 1)
    real_images = torch.clamp(real_images, 0, 1)
    
    # Move to device for computation
    generated_images = generated_images.to(device)
    real_images = real_images.to(device)
    
    # Compute SSIM, SSNR, PSNR, LPIPS (paired comparison between generated and real)
    tqdm.write("Computing SSIM, SSNR, PSNR, LPIPS...")
    ssim_value = float('nan')
    ssnr_value = float('nan')
    psnr_value = float('nan')
    lpips_value = float('nan')
    
    try:
        # Compute metrics by comparing corresponding generated and real images
        # Process in batches to avoid memory issues
        batch_size_metric = 50
        ssim_values = []
        ssnr_values = []
        psnr_values = []
        lpips_values = []
        
        for i in range(0, num_samples, batch_size_metric):
            end_idx = min(i + batch_size_metric, num_samples)
            gen_batch = generated_images[i:end_idx]
            real_batch = real_images[i:end_idx]
            
            # SSIM - returns per-image values when size_average=False
            batch_ssim = ssim(real_batch, gen_batch, size_average=False)
            ssim_values.append(batch_ssim.cpu())
            
            # SSNR - returns single float per batch, compute per image by processing individually
            # For efficiency, we'll compute batch average
            batch_ssnr = ssnr(real_batch, gen_batch)
            ssnr_values.append(batch_ssnr)
            
            # PSNR - returns single float per batch, compute per image by processing individually
            # For efficiency, we'll compute batch average
            batch_psnr = psnr(real_batch, gen_batch)
            psnr_values.append(batch_psnr)
            
            # LPIPS - returns single float per batch
            batch_lpips = lpips(real_batch, gen_batch, device=device)
            lpips_values.append(batch_lpips)
        
        # Average SSIM (it returns per-image values)
        if len(ssim_values) > 0:
            ssim_value = torch.cat(ssim_values).mean().item()
        else:
            ssim_value = float('nan')
        
        # Average other metrics (they return batch averages)
        ssnr_value = np.mean(ssnr_values) if ssnr_values else float('nan')
        psnr_value = np.mean(psnr_values) if psnr_values else float('nan')
        lpips_value = np.mean(lpips_values) if lpips_values else float('nan')
        
    except Exception as e:
        tqdm.write(f"Warning: SSIM/SSNR/PSNR/LPIPS computation failed: {e}")
    
    # Compute FID
    tqdm.write("Computing FID...")
    fid_value = float('nan')
    try:
        fid_value = calculate_fid(
            real_images.cpu(), 
            generated_images.cpu(), 
            device=device, 
            batch_size=50
        )
    except Exception as e:
        tqdm.write(f"Warning: FID computation failed: {e}")
    
    # Compute Inception Score
    tqdm.write("Computing Inception Score...")
    is_mean = float('nan')
    is_std = float('nan')
    try:
        is_mean, is_std = calculate_inception_score(
            generated_images.cpu(), 
            device=device, 
            batch_size=50
        )
    except Exception as e:
        tqdm.write(f"Warning: Inception Score computation failed: {e}")
    
    # Compute Precision and Recall
    tqdm.write("Computing Precision and Recall...")
    precision = float('nan')
    recall = float('nan')
    try:
        precision, recall = calculate_precision_recall(
            real_images.cpu(), 
            generated_images.cpu(), 
            device=device, 
            batch_size=50
        )
    except Exception as e:
        tqdm.write(f"Warning: Precision/Recall computation failed: {e}")
    
    metrics = {
        'ssim': ssim_value,
        'ssnr': ssnr_value,
        'psnr': psnr_value,
        'lpips': lpips_value,
        'fid': fid_value,
        'inception_score_mean': is_mean,
        'inception_score_std': is_std,
        'precision': precision,
        'recall': recall,
    }
    
    tqdm.write(
        f"Generative Metrics - SSIM: {ssim_value:.4f}, SSNR: {ssnr_value:.4f} dB, "
        f"PSNR: {psnr_value:.4f} dB, LPIPS: {lpips_value:.4f}, FID: {fid_value:.4f}, "
        f"IS: {is_mean:.4f} ± {is_std:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
    )
    
    return metrics


def main(args):
    """
    Main training loop for multi-objective variational autoencoder.
    
    Sets up datasets, model, optimizer, scheduler, and multi-task learning aggregator.
    Runs training for the specified number of epochs, evaluates periodically, generates
    samples, and saves checkpoints. Supports various aggregators for multi-task learning
    and logs metrics to wandb if enabled.
    
    Args:
        args: Configuration object containing all training hyperparameters:
            - device: Device to use ('cuda:X' or 'cpu')
            - dataset: Dataset name ('CIFAR10', 'CIFAR100', 'CelebA', 'ImageNet')
            - data_dir: Directory containing datasets
            - save_path: Root directory for saving logs and checkpoints
            - epochs: Number of training epochs
            - batch_size: Training batch size
            - aggregator: Multi-task learning aggregator name (e.g., 'upgrad', 'pcgrad', 'sum')
            - arch: Model architecture name
            - optimizer: Optimizer name ('adam', 'sgd', 'adamw', 'rmsprop')
            - lr: Learning rate
            - scheduler: Learning rate scheduler ('cosine', 'multi_step', or None)
            - use_wandb: Whether to use wandb for logging
            - wandb_project: Wandb project name
            - eval_freq: Frequency of evaluation (every N epochs)
            - save_freq: Frequency of saving samples (every N epochs)
            - num_samples: Number of samples to generate for visualization
            - max_fid_samples: Maximum samples for FID computation
            - And other model/training specific parameters
            
    Returns:
        None
        
    Side Effects:
        - Creates directory structure for saving logs and checkpoints
        - Initializes wandb run if use_wandb is True
        - Saves model checkpoints and generated samples
        - Logs metrics to wandb if enabled
    """
    device = torch.device(args.device)

    train_dataset, test_dataset, input_size = get_dataset(args.dataset, data_dir=args.data_dir, normalize=args.normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    args.dataset_size = len(train_dataset)

    net = get_network(input_size, num_channels=3, args=args, device=device).to(device)
    args.total_params = net.total_trainable_params()
    
    # Register each lambda weight separately in args for easier access
    for loss_name, weight in net.lambda_weights.items():
        attr_name = f"{loss_name}_weight"
        setattr(args, attr_name, weight)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = None
    if args.scheduler is not None:
        if args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.scheduler_lr_min)
        elif args.scheduler == "multi_step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)
        else:
            raise ValueError(f"Scheduler {args.scheduler} not supported")

    aggregator = None
    if args.aggregator is not None:
        agg_name = args.aggregator.lower()
        if agg_name == "upgrad":
            aggregator = UPGrad()
        elif agg_name == "pcgrad":
            aggregator = PCGrad()
        elif agg_name == "mean":
            aggregator = Mean()
        elif agg_name == "aligned_mtl":
            aggregator = AlignedMTL()
        elif agg_name == "imtlg":
            aggregator = IMTLG()
        elif agg_name == "mgda":
            aggregator = MGDA()
        elif agg_name == "cagrad":
            aggregator = CAGrad(c=1.0)
        elif agg_name == "nashmtl":
            aggregator = NashMTL(n_tasks=len(net.objectives))
        elif agg_name == "dualproj":
            aggregator = DualProj()
        elif agg_name == "jd_sum":
            aggregator = Sum()
        elif agg_name == "sum":
            aggregator = "sum"
        else:
            raise ValueError(f"Aggregator {args.aggregator} not supported")
    else:
        args.aggregator = "sum"

    if aggregator is not None and aggregator != "sum":
        aggregator.weighting.register_forward_hook(print_weights)
        aggregator.weighting.register_forward_hook(print_gd_similarity)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_path, args.dataset, args.arch, args.optimizer, args.aggregator, timestamp)
    os.makedirs(os.path.join(save_root, "figures", "generated"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "figures", "reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "checkpoints"), exist_ok=True)

    if args.use_wandb:
        # Use API key from .env if available, otherwise use existing wandb login
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name if args.wandb_name else None,
            config=vars(args),
            dir=save_root,
            tags=args.wandb_tags if args.wandb_tags else None,
        )

    # eval_loss_meters = evaluate(net, train_loader, device=device, args=args)
    # print(
    #     "Initial random loss: "
    #     + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in eval_loss_meters.items())
    # )

    if hasattr(net, "print_model_summary"):
        print(net.print_model_summary())

    objective_keys = net.objectives.keys()
    hv_indicator = build_hv_indicator(objective_keys, args)

    # Track best model
    best_eval_loss = float('inf')
    
    step = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss_meters, step = train_epoch(
            net,
            train_loader=train_loader,
            optimizer=optimizer,
            aggregator=aggregator,
            step=step,
            device=device,
            args=args,
        )

        if hv_indicator is not None:
            train_point = np.array([[train_loss_meters[key].avg for key in objective_keys]])
            train_hv = hv_indicator(train_point)
        else:
            train_hv = float("nan")

        # Format train metrics for display
        train_metric_strs = []
        for key, meter in train_loss_meters.items():
            if key == "codebook_usage_percentage":
                train_metric_strs.append(f"{key}: {meter.avg:.2f}%")
            else:
                train_metric_strs.append(f"{key}: {meter.avg:.6e}")
        
        tqdm.write(
            f" Epoch {epoch}/{args.epochs} - Train - "
            + ", ".join(train_metric_strs)
            + f", HV: {train_hv:.2e}"
        )

        log_dict = {
            "epoch": epoch,
            **{f"train/{key}": meter.avg for key, meter in train_loss_meters.items()},
            "train/hv": train_hv,
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        if (epoch % args.save_freq == 0) or epoch in {1, args.epochs}:

            gen_path = os.path.join(save_root, "figures", "generated", f"epoch_{epoch:03d}_random_samples.pdf")
            generate_random_samples(
                net,
                num_samples=args.num_samples,
                device=device,
                save_path=gen_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

            test_path = os.path.join(save_root, "figures", "reconstructed", f"epoch_{epoch:03d}_test_samples.pdf")
            generate_reconstructed_samples(
                net,
                "test",
                test_loader,
                args.num_samples,
                device,
                save_path=test_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

            train_path = os.path.join(save_root, "figures", "reconstructed", f"epoch_{epoch:03d}_train_samples.pdf")
            generate_reconstructed_samples(
                net,
                "train",
                train_loader,
                args.num_samples,
                device,
                save_path=train_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

        if epoch % args.eval_freq == 0 or epoch in {1, args.epochs}:
            eval_loss_meters = evaluate(net, test_loader, device=device, args=args)

            # Update log_dict with all eval metrics (losses and image quality metrics)
            log_dict.update({f"eval/{key}": meter.avg for key, meter in eval_loss_meters.items()})
            
            # Explicitly add image quality metrics to wandb log dict
            if "ssim" in eval_loss_meters:
                log_dict["eval/ssim"] = eval_loss_meters["ssim"].avg
            if "ssnr" in eval_loss_meters:
                log_dict["eval/ssnr"] = eval_loss_meters["ssnr"].avg
            if "psnr" in eval_loss_meters:
                log_dict["eval/psnr"] = eval_loss_meters["psnr"].avg
            if "lpips" in eval_loss_meters:
                log_dict["eval/lpips"] = eval_loss_meters["lpips"].avg
            if "fid" in eval_loss_meters:
                log_dict["eval/fid"] = eval_loss_meters["fid"].avg

            if hv_indicator is not None:
                eval_point = np.array([[eval_loss_meters[key].avg for key in objective_keys]])
                eval_hv = hv_indicator(eval_point)
                log_dict.update({"eval/hv": eval_hv})

            # Format metrics for display
            metric_strs = []
            for key, meter in eval_loss_meters.items():
                if key in ["ssim", "ssnr", "psnr", "lpips", "fid"]:
                    if key == "fid" and np.isnan(meter.avg):
                        metric_strs.append(f"{key}: N/A")
                    else:
                        metric_strs.append(f"{key}: {meter.avg:.4f}")
                elif key == "codebook_usage_percentage":
                    metric_strs.append(f"{key}: {meter.avg:.2f}%")
                else:
                    metric_strs.append(f"{key}: {meter.avg:.6e}")
            
            tqdm.write(
                f" Epoch {epoch}/{args.epochs} - Eval - "
                + ", ".join(metric_strs)
                + f", HV: {eval_hv:.2e}"
            )

        if args.use_wandb:
            wandb.log(log_dict, step=step)

        if scheduler is not None:
            scheduler.step()
        
        # Save best model based on total eval loss
        current_eval_loss = eval_loss_meters['total_loss'].avg
        if current_eval_loss < best_eval_loss:
            best_eval_loss = current_eval_loss
            best_ckpt = os.path.join(save_root, "checkpoints", "best_checkpoint.pth")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'train_losses': {key: meter.avg for key, meter in train_loss_meters.items()},
                'eval_losses': {key: meter.avg for key, meter in eval_loss_meters.items()},
                'best_eval_loss': best_eval_loss,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, best_ckpt)
            tqdm.write(f" *** Best model saved at epoch {epoch} with eval loss: {best_eval_loss:.6e} ***")

    tqdm.write("Training completed!")
    
    # Save final epoch checkpoint with complete information
    final_ckpt = os.path.join(save_root, "checkpoints", "final_checkpoint.pth")
    checkpoint_data = {
        'epoch': args.epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'train_losses': {key: meter.avg for key, meter in train_loss_meters.items()},
        'eval_losses': {key: meter.avg for key, meter in eval_loss_meters.items()},
        'best_eval_loss': best_eval_loss,
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint_data, final_ckpt)
    tqdm.write(f"Final checkpoint (last epoch) saved to: {final_ckpt}")
    tqdm.write(f"Best checkpoint (eval loss: {best_eval_loss:.6e}) saved to: {os.path.join(save_root, 'checkpoints', 'best_checkpoint.pth')}")

    # Evaluate all generative metrics (SSIM, SSNR, PSNR, LPIPS, FID, IS, Precision, Recall)
    gen_metrics = evaluate_generative_metrics(net, test_loader, device, args)
    
    if args.use_wandb:
        # Log all generative metrics to wandb
        wandb.log({
            "final/ssim": gen_metrics['ssim'],
            "final/ssnr": gen_metrics['ssnr'],
            "final/psnr": gen_metrics['psnr'],
            "final/lpips": gen_metrics['lpips'],
            "final/fid": gen_metrics['fid'],
            "final/inception_score_mean": gen_metrics['inception_score_mean'],
            "final/inception_score_std": gen_metrics['inception_score_std'],
            "final/precision": gen_metrics['precision'],
            "final/recall": gen_metrics['recall'],
        }, step=step)
        
        try:
            wandb.save(final_ckpt)
            best_ckpt_path = os.path.join(save_root, "checkpoints", "best_checkpoint.pth")
            wandb.save(best_ckpt_path)
        except (OSError, PermissionError):
            try:
                artifact = wandb.Artifact("final_model", type="model")
                artifact.add_file(final_ckpt)
                artifact.add_file(os.path.join(save_root, "checkpoints", "best_checkpoint.pth"))
                wandb.log_artifact(artifact)
            except Exception:
                wandb.log({
                    "final_checkpoint_path": final_ckpt,
                    "best_checkpoint_path": os.path.join(save_root, "checkpoints", "best_checkpoint.pth")
                })
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="logs/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--aggregator", "--agg", type=str, default=None)
    parser.add_argument("--arch", type=str, default="vae")
    parser.add_argument("--layer_norm", type=str, default="batch")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--recons_dist", type=str, default="gaussian", choices=["bernoulli", "gaussian", "laplacian"], help="Reconstruction distribution: bernoulli (BCE), gaussian (MSE), or laplacian (L1)")
    parser.add_argument("--recons_reduction", type=str, default="mean", choices=["mean", "sum", "scaled_sum"], help="Loss reduction type: mean (per_pixel_mean), sum (per_image_sum), scaled_sum (total_batch_sum_scaled - MSE only)")
    parser.add_argument("--loss_weights", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", "--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--scheduler_lr_min", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_milestones", type=int, nargs="+", default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--num_embeddings", type=int, default=None)
    parser.add_argument("--anneal_steps", type=int, default=None)
    parser.add_argument("--hv_ref", type=float, nargs="+", default=[1.1, 1.1])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mo-vae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--max_fid_samples", type=int, default=5000, help="Maximum number of samples to use for FID computation")
    parser.add_argument("--max_gen_metrics_samples", type=int, default=5000, help="Maximum number of samples to use for IS, Precision, and Recall computation")

    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    main(args)
