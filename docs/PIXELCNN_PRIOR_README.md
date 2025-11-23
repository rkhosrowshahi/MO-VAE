# PixelCNN Prior for VQ-VAE2

This implementation provides a hierarchical PixelCNN prior for learning the distribution of discrete latent codes in VQ-VAE2, enabling high-quality image generation.

## Overview

VQ-VAE2 learns to encode images into discrete latent codes but doesn't model their distribution. The PixelCNN prior learns:
- **P(z_top)**: Distribution of top-level codes
- **P(z_bottom | z_top)**: Distribution of bottom codes conditioned on top codes

## Architecture

### Components

1. **PixelCNN**: Autoregressive model with masked convolutions
   - Ensures each pixel only depends on previously generated pixels
   - Uses gated residual blocks with tanh/sigmoid activations
   
2. **HierarchicalPixelCNN**: Two-level prior for VQ-VAE2
   - `prior_top`: Models top-level code distribution
   - `prior_bottom`: Models bottom codes conditioned on upsampled top codes

## Usage

### 1. Train VQ-VAE2 (Stage 1)

First, train your VQ-VAE2 model as usual:

```bash
python main.py --arch vq_vae2 \
               --dataset cifar10 \
               --embedding_dim 64 \
               --num_embeddings 512 \
               --epochs 100 \
               --batch_size 128
```

### 2. Train PixelCNN Prior (Stage 2)

Once VQ-VAE2 is trained, train the PixelCNN prior on its discrete codes:

```bash
python train_pixelcnn_prior.py \
    --vqvae2_checkpoint path/to/vqvae2_checkpoint.pth \
    --dataset cifar10 \
    --batch_size 128 \
    --epochs 100 \
    --lr 3e-4 \
    --hidden_channels 128 \
    --num_layers 15 \
    --sample_every 5 \
    --output_dir ./outputs/pixelcnn_prior \
    --use_wandb
```

#### Key Arguments:

- `--vqvae2_checkpoint`: Path to trained VQ-VAE2 model
- `--hidden_channels`: Number of channels in PixelCNN (default: 128)
- `--num_layers`: Number of gated residual blocks (default: 15)
- `--temperature`: Sampling temperature for generation (default: 1.0)
- `--sample_every`: Generate samples every N epochs (default: 5)

### 3. Generate Samples

Generate new images using the trained prior:

```bash
python generate_samples_pixelcnn.py \
    --vqvae2_checkpoint path/to/vqvae2_checkpoint.pth \
    --prior_checkpoint path/to/best_prior.pth \
    --num_samples 100 \
    --temperature 1.0 \
    --batch_size 16 \
    --output_dir ./generated_samples \
    --save_grid \
    --grid_nrow 10
```

#### Temperature Control:

- `temperature=1.0`: Standard sampling (balanced)
- `temperature<1.0`: More conservative, higher quality but less diverse (e.g., 0.8)
- `temperature>1.0`: More diverse but potentially lower quality (e.g., 1.2)

## Programmatic Usage

### Training Prior in Code

```python
from models import VQVAE2, HierarchicalPixelCNN
import torch

# Load pre-trained VQ-VAE2
vqvae2 = VQVAE2(...)
vqvae2.load_state_dict(torch.load('vqvae2.pth'))
vqvae2.eval()

# Create prior
prior = HierarchicalPixelCNN(
    num_embeddings=vqvae2.num_embeddings,
    embedding_dim=vqvae2.embedding_dim,
    hidden_channels=128,
    num_layers=15
)

# Training loop
optimizer = torch.optim.Adam(prior.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for images, _ in dataloader:
        # Extract discrete codes
        with torch.no_grad():
            codes = vqvae2.get_code_indices(images)
            z_top = codes['indices_top']
            z_bottom = codes['indices_bottom']
        
        # Train prior
        optimizer.zero_grad()
        losses = prior.loss_function(z_top, z_bottom)
        losses['total_loss'].backward()
        optimizer.step()
```

### Generating Samples in Code

```python
from models import VQVAE2, HierarchicalPixelCNN
import torch

# Load models
vqvae2 = VQVAE2(...)
vqvae2.load_state_dict(torch.load('vqvae2.pth'))
vqvae2.eval()

prior = HierarchicalPixelCNN(...)
prior.load_state_dict(torch.load('prior.pth'))
prior.eval()

# Generate samples
device = torch.device('cuda')
batch_size = 64
temperature = 1.0

samples = prior.sample_with_vqvae2(
    vqvae2_model=vqvae2,
    batch_size=batch_size,
    device=device,
    temperature=temperature
)

# samples: [B, C, H, W] generated images
```

### Low-Level Code Generation

```python
# Sample discrete codes
z_top, z_bottom = prior.sample(
    batch_size=64,
    top_shape=(vqvae2.latent_spatial_dim_top, vqvae2.latent_spatial_dim_top),
    bottom_shape=(vqvae2.latent_spatial_dim_bottom, vqvae2.latent_spatial_dim_bottom),
    device=device,
    temperature=1.0
)

# Convert to embeddings and decode
with torch.no_grad():
    # Top codes to embeddings
    quant_t = vqvae2.vq_top.embedding(z_top.view(batch_size, -1))
    quant_t = quant_t.view(batch_size, vqvae2.latent_spatial_dim_top,
                           vqvae2.latent_spatial_dim_top, vqvae2.embedding_dim)
    quant_t = quant_t.permute(0, 3, 1, 2)
    
    # Bottom codes to embeddings
    quant_b = vqvae2.vq_bottom.embedding(z_bottom.view(batch_size, -1))
    quant_b = quant_b.view(batch_size, vqvae2.latent_spatial_dim_bottom,
                           vqvae2.latent_spatial_dim_bottom, vqvae2.embedding_dim)
    quant_b = quant_b.permute(0, 3, 1, 2)
    
    # Decode
    images = vqvae2.decode(quant_t, quant_b)
```

## Model Architecture Details

### Masked Convolutions

The PixelCNN uses two types of masked convolutions:

- **Type A**: First layer, excludes center pixel (ensures no information leakage)
- **Type B**: Subsequent layers, includes center pixel

### Gated Residual Blocks

Each block uses gated activation:

```
output = tanh(W_f * x) ⊙ sigmoid(W_g * x)
```

This allows the network to learn complex dependencies.

### Hierarchical Modeling

1. **Top Prior**: Standard PixelCNN for top-level codes
2. **Bottom Prior**: Conditional PixelCNN that takes upsampled top codes as additional input

## Performance Tips

### Training

- **Batch Size**: Larger batches (128-256) work better for stable training
- **Learning Rate**: Start with 3e-4, use cosine annealing
- **Gradient Clipping**: Clip to 1.0 to prevent exploding gradients
- **Epochs**: 100-200 epochs typically sufficient for CIFAR-10/100

### Generation

- **Speed**: Autoregressive generation is slow (serial process)
  - For 8×8 top codes: 64 sequential steps
  - For 16×16 bottom codes: 256 sequential steps
- **Batch Generation**: Generate multiple samples in parallel
- **Temperature**: Start with 1.0 and adjust based on results

### Memory

- **Model Size**: ~10-20M parameters depending on configuration
- **Training**: Requires same memory as VQ-VAE2 inference + prior training
- **Generation**: Can be done on smaller GPUs

## Example Results

With proper training, you should see:
- Sharp, realistic images
- Good diversity at temperature=1.0
- Coherent global structure (from hierarchical modeling)

## Common Issues

### Issue: Samples are blurry
- **Solution**: Increase number of layers or hidden channels in PixelCNN
- **Solution**: Train longer (100+ epochs)
- **Solution**: Check VQ-VAE2 reconstruction quality first

### Issue: Samples lack diversity
- **Solution**: Increase sampling temperature (try 1.1-1.3)
- **Solution**: Check codebook utilization in VQ-VAE2

### Issue: Training is unstable
- **Solution**: Reduce learning rate
- **Solution**: Ensure gradient clipping is enabled
- **Solution**: Check for NaN values in loss

### Issue: Generation is too slow
- **Solution**: This is expected (autoregressive nature)
- **Solution**: Generate in larger batches
- **Solution**: Consider using GPU with more memory for larger batches

## Files

- `models/pixelcnn_prior.py`: Core PixelCNN implementation
- `train_pixelcnn_prior.py`: Training script
- `generate_samples_pixelcnn.py`: Sample generation script
- `models/vq_vae2.py`: Updated with `get_code_indices()` method

## References

- [VQ-VAE2 Paper](https://arxiv.org/abs/1906.00446): Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2"
- [PixelCNN Paper](https://arxiv.org/abs/1606.05328): van den Oord et al., "Conditional Image Generation with PixelCNN Decoders"
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937): Original VQ-VAE paper

## Citation

If you use this implementation, please cite the original VQ-VAE2 paper:

```bibtex
@article{razavi2019generating,
  title={Generating Diverse High-Fidelity Images with VQ-VAE-2},
  author={Razavi, Ali and van den Oord, Aaron and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1906.00446},
  year={2019}
}
```

