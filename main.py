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
    # AlignedMTL,
    NashMTL,
    IMTLG,
    # MGDA,
    PCGrad,
    CAGrad,
    DualProj,
    UPGrad,
    Mean,
    Sum
)
from utils.jd import PNUPGrad, NUPGrad, AlignedMTL, MGDA, COMFORT

import wandb
from pymoo.indicators.hv import HV
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import scienceplots
from torchvision.utils import make_grid

from utils.utils import AverageMeter, get_dataset, set_seed
from utils.metrics import (
    ssim, psnr, lpips,
    calculate_fid, calculate_inception_score, calculate_kid,
    extract_inception_features, fid_from_features, kid_from_features,
    # precision_recall_from_features,  # Commented out: computationally expensive
)
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
    # For training: use per-batch calculation (averaged) to avoid memory issues
    codebook_usage_meter = AverageMeter()

    for images, _ in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss_dict = net.loss_function(images, args=outputs)
        total_loss = loss_dict["total_loss"]

        if total_loss.item() > 1e15:
            tqdm.write(f"Step {step}: EXPLODING: Total loss: {total_loss.item():.6e}, Losses: {loss_dict}")

        # Extract codebook_usage_percentage per batch (for VQ-VAE models)
        # This uses per-batch calculation to avoid memory issues during training
        if "codebook_usage_percentage" in outputs:
            codebook_usage_meter.update(outputs["codebook_usage_percentage"], n=images.size(0))

        # Update global step for hooks before backward
        global _current_step
        _current_step = step + 1
        
        try:
            if aggregator is None or aggregator == "sum":
                total_loss.backward()
            else:
                features = None
                if net.features is not None:
                    features = [outputs[feature] for feature in net.features]

                # Exclude total_loss so MTL aggregator gets only component losses
                component_losses = [v for k, v in loss_dict.items() if k != "total_loss"]
                if isinstance(aggregator, MGDA) or isinstance(aggregator, COMFORT):
                    aggregator.set_losses(torch.stack(component_losses))

                if features is not None:
                    mtl_backward(
                        losses=component_losses,
                        features=features,
                        aggregator=aggregator,
                        retain_graph=True,
                    )
                else:
                    backward(component_losses, aggregator=aggregator)
        except RuntimeError as e:
            # Catch CUDA assertion errors (e.g., from BCE loss with values outside [0,1])
            # This can happen with Aligned-MTL aggregation
            if "CUDA" in str(e) or "cuda" in str(e).lower() or "assert" in str(e).lower():
                tqdm.write(f"Step {step}: CUDA error during backward (likely Aligned-MTL): {str(e)}")
                tqdm.write(f"  Losses: {loss_dict}")
                tqdm.write(f"  Skipping this batch...")
                # Skip optimizer step for this batch
                continue
            else:
                # Re-raise if it's a different RuntimeError
                raise

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
            # Add codebook usage percentage if available (per-batch averaged)
            if codebook_usage_meter.count > 0:
                log_dict["train/codebook_usage_percentage"] = codebook_usage_meter.avg
            wandb.log(log_dict, step=step)

    # Return codebook usage meter if it was tracked (per-batch averaged)
    if codebook_usage_meter.count > 0:
        loss_meters["codebook_usage_percentage"] = codebook_usage_meter
    
    return loss_meters, step


def evaluate(net, data_loader, device, args):
    """
    Evaluate the model on a dataset (losses and codebook usage only).
    
    Computes loss metrics on the provided dataset. Reconstruction metrics
    (rFID, PSNR, SSIM, LPIPS) are computed by evaluate_recon_metrics().
    The model is set to evaluation mode and gradients are disabled.
    
    Args:
        net: The neural network model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on (e.g., 'cuda:0' or 'cpu')
        args: Configuration object containing evaluation settings
            
    Returns:
        dict: Dictionary mapping metric names to AverageMeter objects containing:
            - All loss components (e.g., 'reconstruction_loss', 'kl_loss', 'total_loss')
            - 'codebook_usage_percentage' (if applicable)
    """
    net.eval()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()
    
    # Accumulate unique codebook indices across batches for accurate utilization calculation
    all_codebook_indices = None
    all_codebook_indices_top = None  # For VQVAE2
    all_codebook_indices_bottom = None  # For VQVAE2
    codebook_size = None
    is_vqvae2 = False

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = net(images)
            loss_dict = net.loss_function(images, args=outputs)
            total_loss = loss_dict["total_loss"]

            loss_meters["total_loss"].update(total_loss.item())
            for key, value in loss_dict.items():
                loss_meters[key].update(value.item())

            # Accumulate codebook indices across batches for accurate utilization calculation
            # Handle VQVAE (single codebook) and VQVAE2 (two codebooks)
            if "encoding_inds" in outputs and outputs["encoding_inds"] is not None:
                # Single codebook (VQVAE, GGVQVAE)
                batch_indices = outputs["encoding_inds"].detach().cpu()
                if all_codebook_indices is None:
                    all_codebook_indices = batch_indices
                    # Get codebook size from the model
                    if hasattr(net, 'vq_layer') and hasattr(net.vq_layer, 'K'):
                        codebook_size = net.vq_layer.K
                else:
                    all_codebook_indices = torch.cat([all_codebook_indices, batch_indices], dim=0)
            elif "encoding_inds_top" in outputs and "encoding_inds_bottom" in outputs:
                # Two codebooks (VQVAE2) - track separately
                is_vqvae2 = True
                if outputs["encoding_inds_top"] is not None and outputs["encoding_inds_bottom"] is not None:
                    batch_indices_top = outputs["encoding_inds_top"].detach().cpu()
                    batch_indices_bottom = outputs["encoding_inds_bottom"].detach().cpu()
                    # Track top codebook
                    if all_codebook_indices_top is None:
                        all_codebook_indices_top = batch_indices_top
                        if hasattr(net, 'vq_top') and hasattr(net.vq_top, 'K'):
                            codebook_size = net.vq_top.K
                    else:
                        all_codebook_indices_top = torch.cat([all_codebook_indices_top, batch_indices_top], dim=0)
                    # Track bottom codebook
                    if all_codebook_indices_bottom is None:
                        all_codebook_indices_bottom = batch_indices_bottom
                    else:
                        all_codebook_indices_bottom = torch.cat([all_codebook_indices_bottom, batch_indices_bottom], dim=0)
    
    # Calculate final codebook utilization across all batches
    if is_vqvae2 and all_codebook_indices_top is not None and all_codebook_indices_bottom is not None and codebook_size is not None:
        # VQVAE2: calculate for both codebooks and take average
        unique_indices_top = torch.unique(all_codebook_indices_top)
        unique_indices_bottom = torch.unique(all_codebook_indices_bottom)
        num_used_top = unique_indices_top.size(0)
        num_used_bottom = unique_indices_bottom.size(0)
        usage_top = (num_used_top / codebook_size) * 100.0
        usage_bottom = (num_used_bottom / codebook_size) * 100.0
        codebook_usage_percentage = (usage_top + usage_bottom) / 2.0
        codebook_usage_meter = AverageMeter()
        codebook_usage_meter.update(codebook_usage_percentage)
        loss_meters["codebook_usage_percentage"] = codebook_usage_meter
    elif all_codebook_indices is not None and codebook_size is not None:
        # Single codebook (VQVAE, GGVQVAE)
        unique_indices = torch.unique(all_codebook_indices)
        num_used = unique_indices.size(0)
        codebook_usage_percentage = (num_used / codebook_size) * 100.0
        codebook_usage_meter = AverageMeter()
        codebook_usage_meter.update(codebook_usage_percentage)
        loss_meters["codebook_usage_percentage"] = codebook_usage_meter

    return loss_meters


def _compute_recon_metrics_from_tensors(real_t, recon_t, device, batch_size_metric=128, min_size_for_lpips=64):
    """Compute rFID, PSNR, SSIM, LPIPS from already-collected real and recon tensors."""
    out = {'rfid': float('nan'), 'psnr': float('nan'), 'ssim': float('nan'), 'lpips': float('nan')}
    n = min(real_t.size(0), recon_t.size(0))
    if n == 0:
        return out
    real_t = real_t[:n]
    recon_t = recon_t[:n]
    ssim_vals, psnr_vals, lpips_vals = [], [], []
    img_size = real_t.size(-1)
    for i in range(0, n, batch_size_metric):
        end = min(i + batch_size_metric, n)
        r_batch = real_t[i:end].to(device)
        p_batch = recon_t[i:end].to(device)
        try:
            ssim_vals.append(ssim(r_batch, p_batch, size_average=True).item())
        except Exception:
            pass
        try:
            psnr_vals.append(psnr(r_batch, p_batch))
        except Exception:
            pass
        if img_size >= min_size_for_lpips:
            try:
                lpips_vals.append(lpips(r_batch, p_batch, device=device))
            except Exception:
                pass
    if ssim_vals:
        out['ssim'] = np.mean(ssim_vals)
    if psnr_vals:
        out['psnr'] = np.mean(psnr_vals)
    if lpips_vals:
        out['lpips'] = np.mean(lpips_vals)
    if img_size >= min_size_for_lpips and n >= 2:
        try:
            out['rfid'] = calculate_fid(real_t, recon_t, device=device, batch_size=128)
        except Exception as e:
            tqdm.write(f"Warning: rFID computation failed: {e}")
    return out


def evaluate_with_recon_metrics(net, data_loader, device, args):
    """
    Single pass over the data loader: compute test losses and reconstruction metrics.
    
    More efficient than calling evaluate() and evaluate_recon_metrics() separately,
    as the model is run once per batch and (real, recon) are collected in the same loop.
    
    Returns:
        tuple: (loss_meters, recon_metrics) with same types as evaluate() and evaluate_recon_metrics().
    """
    net.eval()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()
    max_samples = getattr(args, 'max_fid_samples', 5000)
    all_real, all_recon = [], []
    all_codebook_indices = None
    all_codebook_indices_top = None
    all_codebook_indices_bottom = None
    codebook_size = None
    is_vqvae2 = False

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = net(images)
            loss_dict = net.loss_function(images, args=outputs)
            total_loss = loss_dict["total_loss"]
            loss_meters["total_loss"].update(total_loss.item())
            for key, value in loss_dict.items():
                loss_meters[key].update(value.item())

            # Collect real/recon for recon metrics (up to max_samples)
            recons = outputs.get("recons")
            if recons is not None:
                current = sum(x.size(0) for x in all_real)
                take = min(images.size(0), max(0, max_samples - current))
                if take > 0:
                    all_real.append(images[:take].cpu())
                    all_recon.append(recons[:take].cpu())

            # Codebook indices (same as evaluate())
            if "encoding_inds" in outputs and outputs["encoding_inds"] is not None:
                batch_indices = outputs["encoding_inds"].detach().cpu()
                if all_codebook_indices is None:
                    all_codebook_indices = batch_indices
                    if hasattr(net, 'vq_layer') and hasattr(net.vq_layer, 'K'):
                        codebook_size = net.vq_layer.K
                else:
                    all_codebook_indices = torch.cat([all_codebook_indices, batch_indices], dim=0)
            elif "encoding_inds_top" in outputs and "encoding_inds_bottom" in outputs:
                is_vqvae2 = True
                if outputs["encoding_inds_top"] is not None and outputs["encoding_inds_bottom"] is not None:
                    batch_top = outputs["encoding_inds_top"].detach().cpu()
                    batch_bottom = outputs["encoding_inds_bottom"].detach().cpu()
                    if all_codebook_indices_top is None:
                        all_codebook_indices_top = batch_top
                        if hasattr(net, 'vq_top') and hasattr(net.vq_top, 'K'):
                            codebook_size = net.vq_top.K
                    else:
                        all_codebook_indices_top = torch.cat([all_codebook_indices_top, batch_top], dim=0)
                    if all_codebook_indices_bottom is None:
                        all_codebook_indices_bottom = batch_bottom
                    else:
                        all_codebook_indices_bottom = torch.cat([all_codebook_indices_bottom, batch_bottom], dim=0)

    # Codebook usage (same as evaluate())
    if is_vqvae2 and all_codebook_indices_top is not None and all_codebook_indices_bottom is not None and codebook_size is not None:
        u_top = torch.unique(all_codebook_indices_top).size(0)
        u_bottom = torch.unique(all_codebook_indices_bottom).size(0)
        pct = ((u_top + u_bottom) / (2.0 * codebook_size)) * 100.0
        m = AverageMeter()
        m.update(pct)
        loss_meters["codebook_usage_percentage"] = m
    elif all_codebook_indices is not None and codebook_size is not None:
        pct = (torch.unique(all_codebook_indices).size(0) / codebook_size) * 100.0
        m = AverageMeter()
        m.update(pct)
        loss_meters["codebook_usage_percentage"] = m

    # Recon metrics from collected tensors
    if len(all_real) > 0:
        real_t = torch.cat(all_real, dim=0)
        recon_t = torch.cat(all_recon, dim=0)
        recon_metrics = _compute_recon_metrics_from_tensors(real_t, recon_t, device)
    else:
        recon_metrics = {'rfid': float('nan'), 'psnr': float('nan'), 'ssim': float('nan'), 'lpips': float('nan')}

    return loss_meters, recon_metrics


def evaluate_recon_metrics(net, data_loader, device, args):
    """
    Evaluate reconstruction metrics: rFID, PSNR, SSIM, LPIPS.
    
    Runs the model on the data loader to obtain reconstructions, then computes
    reconstruction quality metrics comparing real vs reconstructed images.
    
    Prefer evaluate_with_recon_metrics() when you also need test losses so the
    model is run only once over the data.
    
    Args:
        net: The neural network model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        args: Configuration with max_fid_samples (max samples for rFID and metrics)
        
    Returns:
        dict: 'rfid', 'psnr', 'ssim', 'lpips' (NaN if insufficient data or failure)
    """
    net.eval()
    max_samples = getattr(args, 'max_fid_samples', 5000)
    all_real, all_recon = [], []

    with torch.no_grad():
        for images, _ in data_loader:
            if sum(x.size(0) for x in all_real) >= max_samples:
                break
            images = images.to(device)
            outputs = net(images)
            recons = outputs.get("recons")
            if recons is None:
                continue
            take = min(images.size(0), max_samples - sum(x.size(0) for x in all_real))
            if take <= 0:
                break
            all_real.append(images[:take].cpu())
            all_recon.append(recons[:take].cpu())

    if len(all_real) == 0:
        return {'rfid': float('nan'), 'psnr': float('nan'), 'ssim': float('nan'), 'lpips': float('nan')}
    real_t = torch.cat(all_real, dim=0)
    recon_t = torch.cat(all_recon, dim=0)
    return _compute_recon_metrics_from_tensors(real_t, recon_t, device)


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
    Evaluate generative model metrics on generated samples: gFID, IS, Precision/Recall, KID.
    
    Generates samples from the model and computes:
    - gFID: Fréchet Inception Distance (real vs generated)
    - IS: Inception Score
    - Precision/Recall: Using InceptionV3 features
    - KID: Kernel Inception Distance
    
    ImageNet-pretrained metrics are skipped for images smaller than 64x64.
    
    Args:
        net: The neural network model (must have a sample() method)
        test_loader: DataLoader for test/real images
        device: Device to run evaluation on (e.g., 'cuda:0' or 'cpu')
        args: Configuration with max_gen_metrics_samples
            
    Returns:
        dict: 'gfid', 'inception_score_mean', 'inception_score_std', 'precision', 'recall', 'kid'
    """
    tqdm.write("Evaluating generative metrics (gFID, IS, Precision, Recall, KID)...")
    
    num_samples = args.max_gen_metrics_samples
    
    # Validate num_samples
    if num_samples <= 0:
        tqdm.write(f"Warning: max_gen_metrics_samples is {num_samples}, skipping generative metrics evaluation.")
        return {
            'gfid': float('nan'), 'inception_score_mean': float('nan'), 'inception_score_std': float('nan'),
            'precision': float('nan'), 'recall': float('nan'), 'kid': float('nan')
        }
    
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
            # Check if samples are valid
            if samples is None or samples.numel() == 0:
                tqdm.write(f"Warning: net.sample() returned empty tensor for batch {i}, skipping...")
                continue
            generated_samples.append(samples.cpu())
    
    # Check if we have any generated samples
    if len(generated_samples) == 0:
        tqdm.write("Error: No samples were generated. Cannot compute generative metrics.")
        return {
            'gfid': float('nan'), 'inception_score_mean': float('nan'), 'inception_score_std': float('nan'),
            'precision': float('nan'), 'recall': float('nan'), 'kid': float('nan')
        }
    
    generated_images = torch.cat(generated_samples, dim=0)[:num_samples]
    
    # Validate generated_images
    if generated_images.numel() == 0:
        tqdm.write("Error: Generated images tensor is empty. Cannot compute generative metrics.")
        return {
            'gfid': float('nan'), 'inception_score_mean': float('nan'), 'inception_score_std': float('nan'),
            'precision': float('nan'), 'recall': float('nan'), 'kid': float('nan')
        }
    
    # Collect real images from test dataset
    tqdm.write(f"Collecting {num_samples} real images from test dataset...")
    real_images = []
    collected = 0
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Collecting real images", leave=False):
            if collected >= num_samples:
                break
            take = min(images.size(0), num_samples - collected)
            if take > 0:
                real_images.append(images[:take].cpu())
                collected += take
    
    # Check if we have any real images
    if len(real_images) == 0:
        tqdm.write("Error: No real images were collected from test dataset. Cannot compute generative metrics.")
        return {
            'gfid': float('nan'), 'inception_score_mean': float('nan'), 'inception_score_std': float('nan'),
            'precision': float('nan'), 'recall': float('nan'), 'kid': float('nan')
        }
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Validate real_images
    if real_images.numel() == 0:
        tqdm.write("Error: Real images tensor is empty. Cannot compute generative metrics.")
        return {
            'gfid': float('nan'), 'inception_score_mean': float('nan'), 'inception_score_std': float('nan'),
            'precision': float('nan'), 'recall': float('nan'), 'kid': float('nan')
        }
    
    # Ensure both tensors have the same number of samples
    n = min(generated_images.size(0), real_images.size(0))
    if n < num_samples:
        tqdm.write(f"Warning: Only {n} samples available (requested {num_samples}). Using {n} samples for metrics.")
    generated_images = generated_images[:n]
    real_images = real_images[:n]
    # n is the actual count used for all metric loops below (do not clamp: metrics expect [0,1] or [-1,1] per pipeline and handle internally)

    # Move to device for computation
    generated_images = generated_images.to(device)
    real_images = real_images.to(device)
    
    # Check image size - ImageNet-pretrained models may not work well with very small images
    # Note: CelebA-HQ and ImageNet are typically 256x256, which works fine with ImageNet metrics
    # Regular CelebA is often 64x64, which requires skipping ImageNet-pretrained metrics
    img_size = generated_images.size(-1)  # Assuming square images
    min_size_for_imagenet_metrics = 64  # Minimum size for reliable ImageNet-pretrained metrics
    
    if img_size < min_size_for_imagenet_metrics:
        tqdm.write(f"Warning: Image size ({img_size}x{img_size}) is too small for ImageNet-pretrained metrics.")
        tqdm.write(f"Skipping gFID, IS, Precision/Recall, KID (require >= {min_size_for_imagenet_metrics}x{min_size_for_imagenet_metrics}).")
    elif img_size < 128:
        tqdm.write(f"Note: Image size ({img_size}x{img_size}) is smaller than typical ImageNet size (224x224 or 299x299).")
    elif img_size >= 256:
        tqdm.write(f"Image size ({img_size}x{img_size}) is suitable for all generative metrics.")
    
    # Extract Inception features once and reuse for gFID, KID, Precision/Recall
    gfid_value = float('nan')
    kid_value = float('nan')
    precision = float('nan')
    recall = float('nan')
    if img_size >= min_size_for_imagenet_metrics:
        tqdm.write("Extracting Inception features (shared for gFID, KID)...")
        try:
            real_features = extract_inception_features(real_images, device=device, batch_size=128)
            tqdm.write(f"Real features shape: {real_features.shape}")
            fake_features = extract_inception_features(generated_images, device=device, batch_size=128)
            tqdm.write(f"Fake features shape: {fake_features.shape}")
            if len(real_features) > 0 and len(fake_features) > 0:
                gfid_value = fid_from_features(real_features, fake_features)
                tqdm.write(f"gFID value: {gfid_value}")
                kid_value = kid_from_features(real_features, fake_features)
                tqdm.write(f"KID value: {kid_value}")
                # precision, recall = precision_recall_from_features(real_features, fake_features, k=3)  # Commented out: expensive
                # tqdm.write(f"Precision: {precision}, Recall: {recall}")
        except Exception as e:
            tqdm.write(f"Warning: gFID/KID computation failed: {e}")
    
    # Inception Score (separate pass: needs classifier logits, not features)
    is_mean = float('nan')
    is_std = float('nan')
    if img_size >= min_size_for_imagenet_metrics:
        tqdm.write("Computing Inception Score...")
        try:
            is_mean, is_std = calculate_inception_score(
                generated_images.cpu(),
                device=device,
                batch_size=128
            )
        except Exception as e:
            tqdm.write(f"Warning: Inception Score computation failed: {e}")
    
    metrics = {
        'gfid': gfid_value,
        'inception_score_mean': is_mean,
        'inception_score_std': is_std,
        'precision': precision,
        'recall': recall,
        'kid': kid_value,
    }
    
    tqdm.write(
        f"Generative Metrics - gFID: {gfid_value:.4f}, IS: {is_mean:.4f} ± {is_std:.4f}, "
        f"KID: {kid_value:.4f}"
        # f"Precision: {precision:.4f}, Recall: {recall:.4f}, "  # Commented out
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
    # Most models in this repo use `tanh` outputs for gaussian/laplacian losses, which assumes inputs are normalized to [-1, 1].
    # If `--normalize` is off, datasets are in [0, 1] and training will silently suffer.
    if (not args.normalize) and getattr(args, "recons_dist", None) in {"gaussian", "laplacian"}:
        tqdm.write(
            "Warning: `normalize=false` with `recons_dist` in {gaussian, laplacian}. "
            "Your data will be in [0,1] but many decoders output tanh in [-1,1]. "
            "Consider enabling `--normalize` (mean=0.5, std=0.5) for stable training."
        )

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
        elif args.scheduler == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
        else:
            raise ValueError(f"Scheduler {args.scheduler} not supported")

    aggregator = None
    if args.aggregator is not None:
        agg_name = args.aggregator.lower()
        if agg_name == "upgrad":
            aggregator = UPGrad(norm_eps=args.agg_norm_eps, reg_eps=args.agg_reg_eps)
        elif agg_name == "pcgrad":
            aggregator = PCGrad()
        elif agg_name == "mean":
            aggregator = Mean()
        elif agg_name == "aligned_mtl":
            aggregator = AlignedMTL()
        elif agg_name == "aligned_mtl_median":
            aggregator = AlignedMTL(scale_mode="median")
        elif agg_name == "aligned_mtl_rmse":
            aggregator = AlignedMTL(scale_mode="rmse")
        elif agg_name == "imtlg":
            aggregator = IMTLG()
        elif agg_name == 'mgda':
            aggregator = MGDA(epsilon=args.mgda_epsilon, max_iters=args.mgda_max_iters)
        elif agg_name == 'mgda_ln':
            aggregator = MGDA(epsilon=args.mgda_epsilon, max_iters=args.mgda_max_iters, norm_type='l2')
        elif agg_name == 'mgda_gn':
            aggregator = MGDA(epsilon=args.mgda_epsilon, max_iters=args.mgda_max_iters, norm_type='loss')
        elif agg_name == 'mgda_lgn':
            aggregator = MGDA(epsilon=args.mgda_epsilon, max_iters=args.mgda_max_iters, norm_type='loss+')
        elif agg_name == "cagrad":
            aggregator = CAGrad(c=1.0, norm_eps=args.agg_norm_eps)
        elif agg_name == "nashmtl":
            aggregator = NashMTL(n_tasks=len(net.objectives), update_weights_every=len(train_loader), optim_niter=20)
        elif agg_name == "dualproj":
            aggregator = DualProj(norm_eps=args.agg_norm_eps, reg_eps=args.agg_reg_eps)
        elif agg_name == "jd_sum":
            aggregator = Sum()
        elif agg_name == "nupgrad":
            aggregator = NUPGrad(norm_eps=args.agg_norm_eps, reg_eps=args.agg_reg_eps)
        elif agg_name == "pnupgrad":
            aggregator = PNUPGrad(norm_eps=args.agg_norm_eps, reg_eps=args.agg_reg_eps)
        elif agg_name == "sum":
            aggregator = "sum"
        elif agg_name == "comfort":
            aggregator = COMFORT(
                mgda_norm_type=getattr(args, "comfort_mgda_norm_type", "none"),
                mgda_stable=getattr(args, "comfort_mgda_stable", False),
                mgda_epsilon=args.mgda_epsilon,
                mgda_max_iters=args.mgda_max_iters,
                mgda_min_eigenvalue_eps=getattr(args, "mgda_min_eigenvalue_eps", 1e-10),
                beta_k=getattr(args, "comfort_beta_k", 1.0),
                beta_a=getattr(args, "comfort_beta_a", 1.0),
                beta_l=getattr(args, "comfort_beta_l", 0.01),
                beta_u=getattr(args, "comfort_beta_u", 1.0),
            )
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
            group=args.wandb_group if args.wandb_group else None,
            tags=args.wandb_tags if args.wandb_tags else None,
        )

    # eval_loss_meters = evaluate(net, train_loader, device=device, args=args)
    # print(
    #     "Initial random loss: "
    #     + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in eval_loss_meters.items())
    # )

    if hasattr(net, "print_model_summary") and args.device.endswith("0"):
        print(net.print_model_summary())

    objective_keys = net.objectives.keys()
    hv_indicator = build_hv_indicator(objective_keys, args)

    # Track best model
    best_eval_loss = float('inf')
    
    step = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        if isinstance(aggregator, COMFORT):
            aggregator.set_epoch(epoch, args.epochs)
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

            # Update log_dict with all eval metrics (losses, codebook_usage)
            log_dict.update({f"eval/{key}": meter.avg for key, meter in eval_loss_meters.items()})

            eval_hv = None
            if hv_indicator is not None:
                eval_point = np.array([[eval_loss_meters[key].avg for key in objective_keys]])
                eval_hv = hv_indicator(eval_point)
                log_dict.update({"eval/hv": eval_hv})

            # Format metrics for display
            metric_strs = []
            for key, meter in eval_loss_meters.items():
                if key == "codebook_usage_percentage":
                    metric_strs.append(f"{key}: {meter.avg:.2f}%")
                else:
                    metric_strs.append(f"{key}: {meter.avg:.6e}")
            if eval_hv is not None:
                metric_strs.append(f"HV: {eval_hv:.2e}")
            tqdm.write(
                f" Epoch {epoch}/{args.epochs} - Eval - " + ", ".join(metric_strs)
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

    # Evaluate reconstruction metrics (rFID, PSNR, SSIM, LPIPS) and generative metrics (gFID, IS, Precision, Recall, KID)
    loss_meters, recon_metrics = evaluate_with_recon_metrics(net, test_loader, device, args)
    gen_metrics = evaluate_generative_metrics(net, test_loader, device, args)
    
    if args.use_wandb:
        # Log final loss values
        final_losses = {f"final/eval_{key}": meter.avg for key, meter in loss_meters.items() if hasattr(meter, 'avg')}
        if final_losses:
            wandb.log(final_losses, step=step)
        # Log reconstruction metrics
        wandb.log({
            "final/rfid": recon_metrics['rfid'],
            "final/psnr": recon_metrics['psnr'],
            "final/ssim": recon_metrics['ssim'],
            "final/lpips": recon_metrics['lpips'],
        }, step=step)
        # Log generative metrics
        wandb.log({
            "final/gfid": gen_metrics['gfid'],
            "final/inception_score_mean": gen_metrics['inception_score_mean'],
            "final/inception_score_std": gen_metrics['inception_score_std'],
            # "final/precision": gen_metrics['precision'],   # Commented out
            # "final/recall": gen_metrics['recall'],         # Commented out
            "final/kid": gen_metrics['kid'],
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
    parser.add_argument(
        "--agg_norm_eps",
        "--agg-norm-eps",
        "--norm_eps",
        "--norm-eps",
        type=float,
        default=1e-4,
        help="Aggregator normalization epsilon (used by UpGrad/DualProj/CAGrad/NUPGrad/PNUPGrad).",
    )
    parser.add_argument(
        "--agg_reg_eps",
        "--agg-reg-eps",
        "--reg_eps",
        "--reg-eps",
        type=float,
        default=1e-4,
        help="Aggregator regularization epsilon (used by UpGrad/DualProj/NUPGrad/PNUPGrad).",
    )
    parser.add_argument(
        "--mgda_epsilon",
        "--mgda-epsilon",
        type=float,
        default=1e-5,
        help="MGDA Frank-Wolfe stop threshold epsilon (TorchJD default: 1e-5)",
    )
    parser.add_argument(
        "--mgda_max_iters",
        "--mgda-max-iters",
        type=int,
        default=250,
        help="MGDA Frank-Wolfe maximum iterations (TorchJD default: 250)",
    )
    parser.add_argument(
        "--mgda_min_eigenvalue_eps",
        "--mgda-min-eigenvalue-eps",
        type=float,
        default=1e-10,
        help="COMFORT/StableMGDA: min eigenvalue clamp (default 1e-10)",
    )
    parser.add_argument(
        "--comfort_mgda_norm_type",
        "--comfort-mgda-norm-type",
        type=str,
        default="none",
        choices=["none", "l2", "loss", "loss+"],
        help="COMFORT: MGDA branch norm type (default: none)",
    )
    parser.add_argument(
        "--comfort_mgda_stable",
        "--comfort-mgda-stable",
        action="store_true",
        help="COMFORT: use StableMGDA (eigen regularization)",
    )
    parser.add_argument("--comfort_beta_k", type=float, default=1.0, help="COMFORT: beta schedule steepness")
    parser.add_argument("--comfort_beta_a", type=float, default=1.0, help="COMFORT: beta schedule progress power")
    parser.add_argument("--comfort_beta_l", type=float, default=0.01, help="COMFORT: beta at start of training")
    parser.add_argument("--comfort_beta_u", type=float, default=1.0, help="COMFORT: beta at end of training")
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
    parser.add_argument("--recursive_kld_anneal_steps", type=int, default=25000)
    # Sphere Encoder (arXiv:2602.15030) optional args
    parser.add_argument("--sigma_max_angle_deg", type=float, default=80.0, help="Sphere encoder: max noise angle in degrees (paper: 80 for 32x32, 85 for 256x256)")
    parser.add_argument("--sigma_mix_prob", type=float, default=0.0, help="Sphere encoder: probability to sample alpha from a higher-angle mix band (paper: 0.1)")
    parser.add_argument("--sigma_mix_angle_min_deg", type=float, default=None, help="Sphere encoder: mix band min angle in degrees (e.g., 80)")
    parser.add_argument("--sigma_mix_angle_max_deg", type=float, default=None, help="Sphere encoder: mix band max angle in degrees (e.g., 85)")
    parser.add_argument("--lambda_pix_recon", type=float, default=1.0, help="Sphere encoder: weight for pixel reconstruction loss")
    parser.add_argument("--lambda_pix_con", type=float, default=0.5, help="Sphere encoder: weight for pixel consistency loss")
    parser.add_argument("--lambda_lat_con", type=float, default=0.1, help="Sphere encoder: weight for latent consistency loss")
    # Sphere Encoder ViT (paper architecture)
    parser.add_argument("--patch_size", type=int, default=None, help="ViT patch size (default: 2 for img_size<=32, 8 else)")
    parser.add_argument("--vit_embed_dim", type=int, default=1024, help="Sphere encoder ViT: transformer embed dim")
    parser.add_argument("--vit_depth", type=int, default=24, help="Sphere encoder ViT: num transformer blocks")
    parser.add_argument("--vit_num_heads", type=int, default=16, help="Sphere encoder ViT: num attention heads")
    parser.add_argument("--vit_mixer_depth", type=int, default=2, help="Sphere encoder ViT: MLP-Mixer depth (2 CIFAR, 4 large img)")
    parser.add_argument("--num_classes", type=int, default=0, help="Num classes for conditional generation (0 = unconditional)")
    parser.add_argument("--hv_ref", type=float, nargs="+", default=[1.1, 1.1])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mo-vae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--max_fid_samples", type=int, default=10000, help="Maximum number of samples to use for FID computation")
    parser.add_argument("--max_gen_metrics_samples", type=int, default=10000, help="Maximum number of samples to use for IS, Precision, and Recall computation")

    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    main(args)
