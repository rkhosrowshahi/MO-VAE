# PixelCNN Prior Quick Start Guide

Get started with PixelCNN prior for VQ-VAE2 in 3 simple steps!

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Trained VQ-VAE2 model

## 🚀 Quick Start (3 Steps)

### Step 1: Train VQ-VAE2

If you don't have a trained VQ-VAE2 model yet:

```bash
python main.py \
    --arch vq_vae2 \
    --dataset cifar10 \
    --embedding_dim 64 \
    --num_embeddings 512 \
    --hidden_dims 128 256 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.0002
```

This will save a checkpoint to `logs/CIFAR10/vq_vae2/...`

### Step 2: Train PixelCNN Prior

Train the prior on VQ-VAE2's discrete codes:

```bash
python train_pixelcnn_vqvae2.py \
    --vqvae2_checkpoint logs/CIFAR10/vq_vae2/.../best_model.pth \
    --dataset cifar10 \
    --batch_size 128 \
    --epochs 100 \
    --lr 3e-4 \
    --hidden_channels 128 \
    --num_layers 15 \
    --output_dir ./outputs/pixelcnn_prior \
    --sample_every 5
```

The prior will be saved to `outputs/pixelcnn_prior/checkpoints/best_prior.pth`

**Note:** For VQ-VAE (version 1), use `train_pixelcnn_vqvae.py` instead.

### Step 3: Generate Samples

Generate high-quality images:

```bash
python generate_samples_pixelcnn_vqvae2.py \
    --vqvae2_checkpoint logs/CIFAR10/vq_vae2/.../best_model.pth \
    --prior_checkpoint outputs/pixelcnn_prior/checkpoints/best_prior.pth \
    --num_samples 100 \
    --temperature 1.0 \
    --save_grid \
    --output_dir ./generated_samples
```

**Note:** For VQ-VAE (version 1), use `generate_samples_pixelcnn_vqvae.py` instead.

Samples will be saved to `generated_samples/samples_grid_temp1.00.png`

## 🎯 What to Expect

### Training Time

**VQ-VAE2 Training:**
- CIFAR-10: ~2-3 hours (100 epochs on RTX 3090)
- CIFAR-100: ~2-3 hours (100 epochs on RTX 3090)

**PixelCNN Prior Training:**
- CIFAR-10: ~4-5 hours (100 epochs on RTX 3090)
- CIFAR-100: ~4-5 hours (100 epochs on RTX 3090)

### Generation Speed

- **Per Sample**: ~5-10 seconds (autoregressive generation)
- **Batch of 64**: ~10-15 seconds (parallel batch generation)

### Quality

After proper training:
- ✅ Sharp, high-fidelity images
- ✅ Good diversity
- ✅ Coherent structure
- ✅ Better than VQ-VAE2 random sampling

## 🎨 Advanced Usage

### Adjust Sampling Temperature

```bash
# More conservative (higher quality, less diversity)
python generate_samples_pixelcnn_vqvae2.py ... --temperature 0.8

# Standard
python generate_samples_pixelcnn_vqvae2.py ... --temperature 1.0

# More diverse (lower quality, more variety)
python generate_samples_pixelcnn_vqvae2.py ... --temperature 1.2
```

### Use in Python Code

```python
from models import VQVAE2, HierarchicalPixelCNN
import torch

# Load models
device = torch.device('cuda')
vqvae2 = VQVAE2(...).to(device)
vqvae2.load_state_dict(torch.load('vqvae2.pth'))

prior = HierarchicalPixelCNN(...).to(device)
prior.load_state_dict(torch.load('prior.pth')['model_state_dict'])

# Generate
samples = prior.sample_with_vqvae2(vqvae2, batch_size=64, device=device, temperature=1.0)
# samples: [64, 3, 32, 32] for CIFAR-10
```

## 📊 Monitoring Training

### W&B Integration

Add `--use_wandb` to track training:

```bash
python train_pixelcnn_vqvae2.py \
    ... \
    --use_wandb \
    --wandb_project pixelcnn-prior \
    --wandb_entity your_username
```

This logs:
- Training losses (top, bottom, total)
- Learning rate
- Generated samples every N epochs
- Model parameters

### Check Generated Samples

During training, samples are saved to `output_dir/samples/samples_epoch_XXXX.png`

Watch these to see quality improve over time!

## 🔧 Troubleshooting

### Problem: Samples are blurry

**Solutions:**
1. Train longer (try 200 epochs)
2. Increase model capacity:
   ```bash
   --hidden_channels 256 --num_layers 20
   ```
3. Check VQ-VAE2 reconstruction quality first

### Problem: Loss not decreasing

**Solutions:**
1. Lower learning rate: `--lr 1e-4`
2. Check VQ-VAE2 checkpoint loads correctly
3. Ensure dataset matches VQ-VAE2 training dataset

### Problem: Out of memory

**Solutions:**
1. Reduce batch size: `--batch_size 64`
2. Reduce model size: `--hidden_channels 64 --num_layers 10`
3. Use gradient accumulation (modify training script)

### Problem: Generation too slow

**Expected behavior** - autoregressive models are inherently slow.

**Workarounds:**
1. Generate larger batches: `--batch_size 128`
2. Use faster GPU
3. For inference only, consider model quantization

## 📚 Next Steps

1. **Experiment with hyperparameters**: Try different architectures
2. **Different datasets**: Train on CIFAR-100, CelebA, etc.
3. **Conditional generation**: Extend to class-conditional sampling
4. **Advanced priors**: Try other autoregressive models (Transformer, etc.)

## 🎓 Example Notebook

Check out `examples/pixelcnn_prior_example.py` for a complete working example with:
- Training loop
- Sample generation
- Temperature comparison
- Interpolation demonstrations

## 📖 Full Documentation

For detailed API documentation, architecture details, and advanced usage, see:
- `PIXELCNN_PRIOR_README.md` - Complete documentation
- `models/pixelcnn_prior.py` - Implementation with docstrings

## 🐛 Common Gotchas

1. **Checkpoint format**: Ensure VQ-VAE2 checkpoint has `model_state_dict` key
2. **Dataset consistency**: Use same dataset for VQ-VAE2 and prior training
3. **Device mismatch**: Ensure all models are on same device
4. **Normalization**: Generated images match VQ-VAE2's output range (check tanh vs sigmoid)

## 💡 Tips for Best Results

1. **VQ-VAE2 Quality First**: Ensure VQ-VAE2 reconstructions are good before training prior
2. **Codebook Usage**: Check codebook utilization in VQ-VAE2 (should use most codes)
3. **Training Time**: Don't undertrain! 100+ epochs usually needed
4. **Validation**: Generate samples periodically to monitor quality
5. **Temperature Tuning**: Start at 1.0, then experiment

## 🤝 Need Help?

- Check existing issues in the repository
- Review the full documentation in `PIXELCNN_PRIOR_README.md`
- Examine example code in `examples/pixelcnn_prior_example.py`

Happy generating! 🎉

