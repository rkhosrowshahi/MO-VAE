"""
Training script for PixelCNN Prior on VQ-VAE2 discrete codes.

This script trains a hierarchical PixelCNN to model the distribution of 
discrete latent codes from a pre-trained VQ-VAE2 model, enabling high-quality
image generation.

Usage:
    python train_pixelcnn_prior.py --vqvae2_checkpoint path/to/vqvae2.pth \
                                     --dataset cifar10 \
                                     --batch_size 128 \
                                     --epochs 100
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models import VQVAE2, HierarchicalPixelCNN
from utils.utils import get_dataset, set_seed, AverageMeter
from torchvision.utils import save_image, make_grid


def parse_args():
    parser = argparse.ArgumentParser(description='Train PixelCNN Prior for VQ-VAE2')
    
    # Model paths
    parser.add_argument('--vqvae2_checkpoint', type=str, required=True,
                        help='Path to pre-trained VQ-VAE2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/pixelcnn_prior',
                        help='Directory to save prior checkpoints')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'],
                        help='Dataset to train on')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing dataset')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    
    # PixelCNN architecture
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Number of hidden channels in PixelCNN')
    parser.add_argument('--num_layers', type=int, default=15,
                        help='Number of residual layers in PixelCNN')
    
    # Sampling
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate samples every N epochs')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='pixelcnn_vqvae2',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default="rasa_research",
                        help='W&B entity name')
    
    return parser.parse_args()


def load_vqvae2(checkpoint_path, device):
    """Load pre-trained VQ-VAE2 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments from checkpoint
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        args = checkpoint.get('args', None)
    else:
        model_state = checkpoint
        args = None
    
    # Infer architecture from state dict
    # This is a simplified version - you may need to adapt based on your checkpoint format
    sample_key = list(model_state.keys())[0]
    
    # Try to extract parameters from state dict
    if 'vq_top.embedding.weight' in model_state:
        num_embeddings = model_state['vq_top.embedding.weight'].shape[0]
        embedding_dim = model_state['vq_top.embedding.weight'].shape[1]
    else:
        raise ValueError("Cannot infer VQ-VAE2 parameters from checkpoint")
    
    # Create model (you may need to adjust these parameters)
    model = VQVAE2(
        in_channels=3,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        hidden_dims=[128, 256],  # Adjust as needed
        input_size=32,  # Adjust based on your dataset
        device=device
    )
    
    # Load weights
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    
    # Freeze VQ-VAE2
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Loaded VQ-VAE2 with {num_embeddings} embeddings of dimension {embedding_dim}")
    print(f"Top latent size: {model.latent_spatial_dim_top}x{model.latent_spatial_dim_top}")
    print(f"Bottom latent size: {model.latent_spatial_dim_bottom}x{model.latent_spatial_dim_bottom}")
    
    return model


def train_epoch(prior, vqvae2, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    prior.train()
    
    loss_top_meter = AverageMeter()
    loss_bottom_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Extract discrete codes from VQ-VAE2
        with torch.no_grad():
            code_dict = vqvae2.get_code_indices(images)
            z_top = code_dict['indices_top']
            z_bottom = code_dict['indices_bottom']
        
        # Forward pass through prior
        optimizer.zero_grad()
        loss_dict = prior.loss_function(z_top, z_bottom)
        
        loss_top = loss_dict['loss_top']
        loss_bottom = loss_dict['loss_bottom']
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        
        optimizer.step()
        
        # Update meters
        loss_top_meter.update(loss_top.item(), images.size(0))
        loss_bottom_meter.update(loss_bottom.item(), images.size(0))
        loss_total_meter.update(total_loss.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss_top': f'{loss_top_meter.avg:.4f}',
            'loss_bottom': f'{loss_bottom_meter.avg:.4f}',
            'total': f'{loss_total_meter.avg:.4f}'
        })
    
    return {
        'loss_top': loss_top_meter.avg,
        'loss_bottom': loss_bottom_meter.avg,
        'total_loss': loss_total_meter.avg
    }


@torch.no_grad()
def generate_samples(prior, vqvae2, num_samples, device, temperature=1.0):
    """Generate samples using the prior and VQ-VAE2 decoder."""
    prior.eval()
    vqvae2.eval()
    
    samples = prior.sample_with_vqvae2(
        vqvae2_model=vqvae2,
        batch_size=num_samples,
        device=device,
        temperature=temperature
    )
    
    return samples


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Load pre-trained VQ-VAE2
    print(f"Loading VQ-VAE2 from {args.vqvae2_checkpoint}")
    vqvae2 = load_vqvae2(args.vqvae2_checkpoint, device)
    
    # Create PixelCNN prior
    print("Creating HierarchicalPixelCNN prior...")
    prior = HierarchicalPixelCNN(
        num_embeddings=vqvae2.num_embeddings,
        embedding_dim=vqvae2.embedding_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Prior has {prior.total_trainable_params():,} trainable parameters")
    
    # Setup optimizer
    optimizer = optim.Adam(
        prior.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, _ = get_dataset(args.dataset, args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        metrics = train_epoch(prior, vqvae2, train_loader, optimizer, device, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Loss Top: {metrics['loss_top']:.4f}, "
              f"Loss Bottom: {metrics['loss_bottom']:.4f}, "
              f"Total: {metrics['total_loss']:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss_top': metrics['loss_top'],
                'train/loss_bottom': metrics['loss_bottom'],
                'train/total_loss': metrics['total_loss'],
                'train/lr': optimizer.param_groups[0]['lr']
            })
        
        # Generate samples
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            print(f"Generating {args.num_samples} samples...")
            samples = generate_samples(
                prior, vqvae2, args.num_samples, device, args.temperature
            )
            
            # Save samples
            sample_path = samples_dir / f'samples_epoch_{epoch:04d}.png'
            save_image(
                samples,
                sample_path,
                nrow=8,
                normalize=True,
                value_range=(-1, 1) if vqvae2.output_activation.__class__.__name__ == 'Tanh' else (0, 1)
            )
            print(f"Saved samples to {sample_path}")
            
            if args.use_wandb:
                wandb.log({
                    'samples': wandb.Image(str(sample_path)),
                    'epoch': epoch
                })
        
        # Save checkpoint
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            checkpoint_path = checkpoints_dir / 'best_prior.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': prior.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoint_path = checkpoints_dir / f'prior_epoch_{epoch:04d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': prior.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': metrics['total_loss'],
                'args': vars(args)
            }, checkpoint_path)
    
    # Final checkpoint
    final_checkpoint_path = checkpoints_dir / 'final_prior.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': metrics['total_loss'],
        'args': vars(args)
    }, final_checkpoint_path)
    
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint saved to {final_checkpoint_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

