# MO-VAE

A multi-objective representation learning approach for variational autoencoders (VAE, Beta-TC-VAE, VQ-VAE, VQ-VAE2, GG-VAE, GG-VQ-VAE) that stabilizes gradients by decomposing the evidence lower bound (ELBO) into complementary objectives.
We leverage multi-task gradient aggregation strategies—`sum`, `UPGrad`, `MGDA`, `Aligned-MTL`—to jointly optimize reconstruction error and latent space regularization terms while keeping gradient updates conflict-free.

## Results

### CelebA VQ-VAE

| Sum (Baseline) | UPGrad | AlignedMTL |
|:--------------:|:------:|:----------:|
| ![Sum](figures/celeba/vq_vae/sum.png) | ![UPGrad](figures/celeba/vq_vae/upgrad.png) | ![AMTL](figures/celeba/vq_vae/amtl.png) |

### CIFAR100 VQ-VAE

| Sum (Baseline) | UPGrad | AlignedMTL |
|:--------------:|:------:|:----------:|
| ![Sum](figures/cifar100/vq_vae/sum.png) | ![UPGrad](figures/cifar100/vq_vae/upgrad.png) | ![AMTL](figures/cifar100/vq_vae/amtl.png) |

## Requirements

- Python 3.10+
- CUDA 12.8+ (for GPU support)
- 8GB+ GPU memory recommended

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/rkhosroshahi/MO-VAE
cd MO-VAE
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
```

Activate the environment:
- **Linux/macOS**: `source .venv/bin/activate`
- **Windows**: `.venv\Scripts\activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note**: The requirements include PyTorch with CUDA 12.8. If you have a different CUDA version, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/).

## Setup

### Datasets

Datasets are automatically downloaded to `./data/` on first run. Supported datasets:
- **CIFAR10 / CIFAR100**: 32×32 color images
- **CelebA / CelebA-HQ**: celebrity face images (64×64)
- **Oxford-Flower-102**: flower images (64×64)
- **animal-face**: AFHQ-style animal faces
- **imagenet**: ImageNet (224×224)

To use a custom data directory:
```bash
python main.py --data_dir /path/to/data ...
```

### Weights & Biases (WandB)

This project uses [WandB](https://wandb.ai) for experiment tracking.

#### 1. Create a WandB account
Sign up at [wandb.ai](https://wandb.ai/signup)

#### 2. Login to WandB
```bash
wandb login
```
Enter your API key when prompted (find it at [wandb.ai/authorize](https://wandb.ai/authorize)).

#### 3. WandB arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--use_wandb` | Enable WandB logging | `False` |
| `--wandb_project` | Project name | `mo-vae` |
| `--wandb_entity` | Team/username | `None` (personal) |
| `--wandb_name` | Run name | `None` (auto-generated) |
| `--wandb_tags` | Tags for filtering | `None` |

#### Training without WandB
Omit the `--use_wandb` flag:
```bash
python main.py --dataset cifar100 --arch vae --epochs 100 --agg sum ...
```

### Configuration Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset (`cifar10`, `cifar100`, `celeba`, `celeba-hq`, `oxford-flower-102`, `animal-face`, `imagenet`) | `CIFAR10` |
| `--arch` | Architecture (`vae`, `betatc_vae`, `vq_vae`, `vq_vae2`, `gg_vae`, `gg_vq_vae`, `gg_vq_vae_v3`, `gg_vq_vae2`) | `vae` |
| `--agg` | Aggregator (`sum`, `upgrad`, `mgda_ln`, `mgda_gn`, `mgda_lgn`, `aligned_mtl`, `aligned_mtl_median`) | `sum` |
| `--epochs` | Training epochs | `50` |
| `--batch_size` | Batch size | `128` |
| `--lr` | Learning rate | `0.001` |
| `--optimizer` | Optimizer (`adam`, `adamw`, `sgd`) | `adam` |
| `--latent_dim` | Latent dimension (VAE, Beta-TC-VAE) | `128` |
| `--hidden_dims` | Encoder/decoder channels | `[32,64,128,256,512]` |
| `--embedding_dim` | VQ-VAE embedding dimension | `64` |
| `--num_embeddings` | VQ-VAE codebook size | `512` |
| `--recons_objective` | Reconstruction loss (`mse`, `bce`, `l1`, `smooth_l1`, `perceptual`) | `mse` |
| `--recons_activation` | Decoder activation (`tanh`, `sigmoid`, `none`) | Inferred from recons_objective |
| `--normalize_inputs` | Normalize inputs to [-1,1] (use with mse/l1 objectives) | `False` |
| `--num_vis_samples` | Samples per visualization grid | `4` |
| `--device` | Device (`cuda:0`, `cpu`) | Auto-detect |
| `--save_path` | Output directory | `logs/` |
| `--save_freq` | Save samples every N epochs | `10` |
| `--eval_freq` | Evaluate every N epochs | `1` |
| `--seed` | Random seed for reproducibility | `None` |

### YAML Configs & Runner

Configs are stored under `configs/<dataset>/<arch>/<aggregator>/<recons_objective>/`. Run via:
```bash
# Single config
python runner.py --f configs/oxford-flower-102/vq_vae2/sum/bce/config_1.yaml

# From file list (bce or mse configs)
python runner.py --file-list configs/oxford-flower-102/vq_vae2/bce_configs.txt
python runner.py --file-list configs/oxford-flower-102/vq_vae2/mse_configs.txt
```

## Usage

### VAE

```bash
# Train VAE on CIFAR100 with sum aggregator
python main.py --dataset cifar100 --arch vae --epochs 100 --agg sum --latent_dim 128 --recons_objective bce --recons_activation sigmoid --use_wandb --wandb_name "VAE CIFAR100 sum"

# Train VAE on CelebA with UPGrad
python main.py --dataset celeba --arch vae --epochs 100 --agg upgrad --latent_dim 128 --recons_objective bce --recons_activation sigmoid --use_wandb --wandb_name "VAE CelebA upgrad"
```

### Beta TC-VAE

```bash
python main.py --dataset cifar100 --arch betatc_vae --epochs 100 --agg upgrad --latent_dim 128 --recons_objective bce --recons_activation sigmoid --use_wandb --wandb_name "BetaTCVAE CIFAR100 upgrad"
```

### VQ-VAE / VQ-VAE2 / GG-VQ-VAE

```bash
# VQ-VAE with BCE
python main.py --dataset oxford-flower-102 --arch vq_vae --epochs 100 --agg upgrad --embedding_dim 64 --num_embeddings 512 --recons_objective bce --recons_activation sigmoid --use_wandb

# VQ-VAE2 (hierarchical)
python main.py --dataset oxford-flower-102 --arch vq_vae2 --epochs 100 --agg sum --embedding_dim 64 --num_embeddings 512 --recons_objective bce --recons_activation sigmoid --use_wandb

# GG-VQ-VAE2 (VQ-VAE2 + gradient-guided + edge-matching losses)
python main.py --dataset oxford-flower-102 --arch gg_vq_vae2 --epochs 100 --agg sum --embedding_dim 64 --num_embeddings 512 --recons_objective bce --recons_activation sigmoid --use_wandb
```

For MSE objectives, enable input normalization:
```bash
python main.py --dataset oxford-flower-102 --arch vq_vae2 --recons_objective mse --recons_activation tanh --normalize_inputs ...
```
<!-- CONTACT -->
## Contact
Rasa Khosrowshahli - rkhosrowshahli@brocku.ca

## Citation
If you find this repository helpful, please cite it as:
```
@misc{khosrowshahli2025movae,
  title        = {Multi-Objective Variational Autoencoders},
  author       = {Rasa Khosrowshahli},
  year         = {2025},
  howpublished = {\url{https://github.com/rkhosroshahi/MO-VAE}},
}
```