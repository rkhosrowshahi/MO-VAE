"""
Training script for PixelCNN Prior on VQ-VAE (version 1) discrete codes.

This is simpler than VQ-VAE2 since there's only one level of codes.

Usage:
    python train_pixelcnn_vqvae.py --vqvae_checkpoint path/to/vqvae.pth \
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

from models import VQVAE, PixelCNN  # Note: Using single-level PixelCNN, not Hierarchical
from utils.utils import get_dataset, set_seed, AverageMeter
from torchvision.utils import save_image, make_grid


def parse_args():
    parser = argparse.ArgumentParser(description='Train PixelCNN Prior for VQ-VAE')
    
    # Model paths
    parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                        help='Path to pre-trained VQ-VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/pixelcnn_vqvae',
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
    parser.add_argument('--wandb_project', type=str, default='pixelcnn_vqvae',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default="rasa_research",
                        help='W&B entity name')
    
    return parser.parse_args()


def load_vqvae(checkpoint_path, device):
    """Load pre-trained VQ-VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments from checkpoint
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        args = checkpoint.get('args', None)
    else:
        model_state = checkpoint
        args = None
    
    # Infer architecture from state dict
    if 'vq_layer.embedding.weight' in model_state:
        num_embeddings = model_state['vq_layer.embedding.weight'].shape[0]
        embedding_dim = model_state['vq_layer.embedding.weight'].shape[1]
    else:
        raise ValueError("Cannot infer VQ-VAE parameters from checkpoint")
    
    # Create model (you may need to adjust these parameters)
    model = VQVAE(
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
    
    # Freeze VQ-VAE
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Loaded VQ-VAE with {num_embeddings} embeddings of dimension {embedding_dim}")
    print(f"Latent size: {model.latent_spatial_dim}×{model.latent_spatial_dim}")
    
    return model


def train_epoch(prior, vqvae, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    prior.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Extract discrete codes from VQ-VAE
        with torch.no_grad():
            z = vqvae.get_code_indices(images)  # [B, H, W]
        
        # Forward pass through prior
        optimizer.zero_grad()
        logits = prior(z)  # [B, num_embeddings, H, W]
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, vqvae.num_embeddings),
            z.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return {'loss': loss_meter.avg}


@torch.no_grad()
def generate_samples(prior, vqvae, num_samples, device, temperature=1.0):
    """Generate samples using the prior and VQ-VAE decoder."""
    prior.eval()
    vqvae.eval()
    
    # Sample codes from prior
    z = prior.sample(
        batch_size=num_samples,
        height=vqvae.latent_spatial_dim,
        width=vqvae.latent_spatial_dim,
        device=device,
        temperature=temperature
    )  # [B, H, W]
    
    # Convert to embeddings
    z_flat = z.view(num_samples, -1)  # [B, H*W]
    quantized = vqvae.vq_layer.embedding(z_flat)  # [B, H*W, D]
    
    # Reshape to spatial
    quantized = quantized.view(
        num_samples,
        vqvae.latent_spatial_dim,
        vqvae.latent_spatial_dim,
        vqvae.embedding_dim
    )
    quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
    
    # Decode
    samples = vqvae.decode(quantized)
    
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
    
    # Load pre-trained VQ-VAE
    print(f"Loading VQ-VAE from {args.vqvae_checkpoint}")
    vqvae = load_vqvae(args.vqvae_checkpoint, device)
    
    # Create PixelCNN prior (single level, not hierarchical)
    print("Creating PixelCNN prior...")
    prior = PixelCNN(
        num_embeddings=vqvae.num_embeddings,
        embedding_dim=vqvae.embedding_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    ).to(device)
    
    num_params = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    print(f"Prior has {num_params:,} trainable parameters")
    
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
        metrics = train_epoch(prior, vqvae, train_loader, optimizer, device, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Loss: {metrics['loss']:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': metrics['loss'],
                'train/lr': optimizer.param_groups[0]['lr']
            })
        
        # Generate samples
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            print(f"Generating {args.num_samples} samples...")
            samples = generate_samples(
                prior, vqvae, args.num_samples, device, args.temperature
            )
            
            # Save samples
            sample_path = samples_dir / f'samples_epoch_{epoch:04d}.png'
            save_image(
                samples,
                sample_path,
                nrow=8,
                normalize=True,
                value_range=(-1, 1) if vqvae.output_activation.__class__.__name__ == 'Tanh' else (0, 1)
            )
            print(f"Saved samples to {sample_path}")
            
            if args.use_wandb:
                wandb.log({
                    'samples': wandb.Image(str(sample_path)),
                    'epoch': epoch
                })
        
        # Save checkpoint
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
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
                'loss': metrics['loss'],
                'args': vars(args)
            }, checkpoint_path)
    
    # Final checkpoint
    final_checkpoint_path = checkpoints_dir / 'final_prior.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': prior.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': metrics['loss'],
        'args': vars(args)
    }, final_checkpoint_path)
    
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint saved to {final_checkpoint_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

