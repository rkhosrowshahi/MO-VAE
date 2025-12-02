"""
Generate samples using trained PixelCNN prior and VQ-VAE (version 1).

Usage:
    python generate_samples_pixelcnn_vqvae.py \
        --vqvae_checkpoint path/to/vqvae.pth \
        --prior_checkpoint path/to/prior.pth \
        --num_samples 100 \
        --temperature 1.0 \
        --output_dir ./generated_samples
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models import VQVAE, PixelCNN


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples with PixelCNN Prior for VQ-VAE')
    
    parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--prior_checkpoint', type=str, required=True,
                        help='Path to trained PixelCNN prior checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher=more diverse)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                        help='Directory to save generated images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_grid', action='store_true',
                        help='Save samples as a grid image')
    parser.add_argument('--grid_nrow', type=int, default=10,
                        help='Number of images per row in grid')
    
    return parser.parse_args()


def load_models(vqvae_path, prior_path, device):
    """Load VQ-VAE and PixelCNN prior from checkpoints."""
    
    # Load VQ-VAE
    print(f"Loading VQ-VAE from {vqvae_path}")
    vqvae_checkpoint = torch.load(vqvae_path, map_location=device)
    
    if 'model_state_dict' in vqvae_checkpoint:
        vqvae_state = vqvae_checkpoint['model_state_dict']
    else:
        vqvae_state = vqvae_checkpoint
    
    # Infer parameters
    num_embeddings = vqvae_state['vq_layer.embedding.weight'].shape[0]
    embedding_dim = vqvae_state['vq_layer.embedding.weight'].shape[1]
    
    vqvae = VQVAE(
        in_channels=3,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        hidden_dims=[128, 256],
        input_size=32,
        device=device
    )
    vqvae.load_state_dict(vqvae_state, strict=False)
    vqvae.to(device)
    vqvae.eval()
    
    print(f"VQ-VAE loaded: {num_embeddings} embeddings, dim={embedding_dim}")
    print(f"Latent size: {vqvae.latent_spatial_dim}×{vqvae.latent_spatial_dim}")
    
    # Load PixelCNN prior
    print(f"Loading PixelCNN prior from {prior_path}")
    prior_checkpoint = torch.load(prior_path, map_location=device)
    
    if 'model_state_dict' in prior_checkpoint:
        prior_state = prior_checkpoint['model_state_dict']
        prior_args = prior_checkpoint.get('args', {})
    else:
        prior_state = prior_checkpoint
        prior_args = {}
    
    prior = PixelCNN(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        hidden_channels=prior_args.get('hidden_channels', 128),
        num_layers=prior_args.get('num_layers', 15)
    )
    prior.load_state_dict(prior_state)
    prior.to(device)
    prior.eval()
    
    num_params = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    print(f"PixelCNN prior loaded: {num_params:,} parameters")
    
    return vqvae, prior


@torch.no_grad()
def generate_samples(vqvae, prior, num_samples, batch_size, device, temperature):
    """Generate samples in batches."""
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples with temperature={temperature}...")
    
    for i in tqdm(range(num_batches), desc='Generating'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Sample discrete codes from prior
        z = prior.sample(
            batch_size=current_batch_size,
            height=vqvae.latent_spatial_dim,
            width=vqvae.latent_spatial_dim,
            device=device,
            temperature=temperature
        )
        
        # Convert to embeddings
        z_flat = z.view(current_batch_size, -1)
        quantized = vqvae.vq_layer.embedding(z_flat)
        
        # Reshape to spatial
        quantized = quantized.view(
            current_batch_size,
            vqvae.latent_spatial_dim,
            vqvae.latent_spatial_dim,
            vqvae.embedding_dim
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        samples = vqvae.decode(quantized)
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    vqvae, prior = load_models(
        args.vqvae_checkpoint,
        args.prior_checkpoint,
        device
    )
    
    # Generate samples
    samples = generate_samples(
        vqvae, prior,
        args.num_samples,
        args.batch_size,
        device,
        args.temperature
    )
    
    # Determine value range based on VQ-VAE output activation
    value_range = (-1, 1) if vqvae.output_activation.__class__.__name__ == 'Tanh' else (0, 1)
    
    # Save samples
    if args.save_grid:
        # Save as grid
        grid_path = output_dir / f'samples_grid_temp{args.temperature:.2f}.png'
        save_image(
            samples,
            grid_path,
            nrow=args.grid_nrow,
            normalize=True,
            value_range=value_range
        )
        print(f"Saved grid to {grid_path}")
    else:
        # Save individual images
        individual_dir = output_dir / f'individual_temp{args.temperature:.2f}'
        individual_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(tqdm(samples, desc='Saving')):
            sample_path = individual_dir / f'sample_{i:05d}.png'
            save_image(
                sample,
                sample_path,
                normalize=True,
                value_range=value_range
            )
        
        print(f"Saved {len(samples)} individual images to {individual_dir}")
    
    print("Generation complete!")


if __name__ == '__main__':
    main()

