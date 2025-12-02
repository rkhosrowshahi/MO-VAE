"""
Generate samples using trained PixelCNN prior and VQ-VAE2.

Usage:
    python generate_samples_pixelcnn.py --vqvae2_checkpoint path/to/vqvae2.pth \
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

from models import VQVAE2, HierarchicalPixelCNN


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples with PixelCNN Prior')
    
    parser.add_argument('--vqvae2_checkpoint', type=str, required=True,
                        help='Path to VQ-VAE2 checkpoint')
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


def load_models(vqvae2_path, prior_path, device):
    """Load VQ-VAE2 and PixelCNN prior from checkpoints."""
    
    # Load VQ-VAE2
    print(f"Loading VQ-VAE2 from {vqvae2_path}")
    vqvae2_checkpoint = torch.load(vqvae2_path, map_location=device)
    
    if 'model_state_dict' in vqvae2_checkpoint:
        vqvae2_state = vqvae2_checkpoint['model_state_dict']
    else:
        vqvae2_state = vqvae2_checkpoint
    
    # Infer parameters
    num_embeddings = vqvae2_state['vq_top.embedding.weight'].shape[0]
    embedding_dim = vqvae2_state['vq_top.embedding.weight'].shape[1]
    
    vqvae2 = VQVAE2(
        in_channels=3,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        hidden_dims=[128, 256],
        input_size=32,
        device=device
    )
    vqvae2.load_state_dict(vqvae2_state, strict=False)
    vqvae2.to(device)
    vqvae2.eval()
    
    print(f"VQ-VAE2 loaded: {num_embeddings} embeddings, dim={embedding_dim}")
    
    # Load PixelCNN prior
    print(f"Loading PixelCNN prior from {prior_path}")
    prior_checkpoint = torch.load(prior_path, map_location=device)
    
    if 'model_state_dict' in prior_checkpoint:
        prior_state = prior_checkpoint['model_state_dict']
        prior_args = prior_checkpoint.get('args', {})
    else:
        prior_state = prior_checkpoint
        prior_args = {}
    
    prior = HierarchicalPixelCNN(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        hidden_channels=prior_args.get('hidden_channels', 128),
        num_layers=prior_args.get('num_layers', 15)
    )
    prior.load_state_dict(prior_state)
    prior.to(device)
    prior.eval()
    
    print(f"PixelCNN prior loaded: {prior.total_trainable_params():,} parameters")
    
    return vqvae2, prior


@torch.no_grad()
def generate_samples(vqvae2, prior, num_samples, batch_size, device, temperature):
    """Generate samples in batches."""
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples with temperature={temperature}...")
    
    for i in tqdm(range(num_batches), desc='Generating'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        samples = prior.sample_with_vqvae2(
            vqvae2_model=vqvae2,
            batch_size=current_batch_size,
            device=device,
            temperature=temperature
        )
        
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
    vqvae2, prior = load_models(
        args.vqvae2_checkpoint,
        args.prior_checkpoint,
        device
    )
    
    # Generate samples
    samples = generate_samples(
        vqvae2, prior,
        args.num_samples,
        args.batch_size,
        device,
        args.temperature
    )
    
    # Determine value range based on VQ-VAE2 output activation
    value_range = (-1, 1) if vqvae2.output_activation.__class__.__name__ == 'Tanh' else (0, 1)
    
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

