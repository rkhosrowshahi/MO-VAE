"""
Training script for PixelCNN Prior on VQ-VAE2 and GG-VQ-VAE2 discrete codes.

VQ-VAE2 / GG-VQ-VAE2 checkpoint can be loaded from:
  - wandb run (--wandb_id) - downloads via wandb API
  - Local path (--vqvae2_checkpoint) - use existing file

After training, optionally resumes the wandb run and:
  - Replaces final/* generative metrics (gfid, inception_score, kid)
  - Uploads prior checkpoint
  - Uploads random_prior_generated_samples images

Usage:
    # From wandb run
    python train_prior_vqvae2.py --wandb_id c3hamj3d --wandb_project mo-vae --wandb_entity rasa_research \
                                --dataset cifar10 --epochs 100 --use_wandb

    # From local checkpoint
    python train_prior_vqvae2.py --vqvae2_checkpoint path/to/vqvae2.pth --dataset cifar10 --epochs 100
"""

import fnmatch
import os
import argparse
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models import get_network, VQVAE2, HierarchicalPixelCNN
from utils.utils import get_dataset, set_seed, AverageMeter
from utils.vq_codes_lmdb import get_or_extract_codes_lmdb
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Train PixelCNN Prior for VQ-VAE2')
    
    # Model paths: either wandb_id (download) OR vqvae2_checkpoint (local)
    parser.add_argument('--wandb_id', type=str, default=None,
                        help='WandB run ID (e.g. c3hamj3d) to download VQ-VAE2 checkpoint from')
    parser.add_argument('--vqvae2_checkpoint', type=str, default=None,
                        help='Local path to pre-trained VQ-VAE2 checkpoint (overrides wandb_id if set)')
    
    parser.add_argument('--output_dir', type=str, default='./outputs/pixelcnn_vqvae2',
                        help='Directory to save prior checkpoints')
    
    # WandB (for download + resume)
    parser.add_argument('--wandb_project', type=str, default='mo-vae',
                        help='W&B project (for downloading and resuming run)')
    parser.add_argument('--wandb_entity', type=str, default='rasa_research',
                        help='W&B entity')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name (e.g. cifar10, cifar100, celeba, imagenet, oxford-flower-102)')
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
    
    # LMDB pre-extracted codes
    parser.add_argument('--prior_use_lmdb_codes', action='store_true', default=True,
                        help='Use LMDB pre-extracted codes (default)')
    parser.add_argument('--no_prior_lmdb_codes', action='store_false', dest='prior_use_lmdb_codes',
                        help='Extract codes on-the-fly each batch')
    parser.add_argument('--prior_force_extract_codes', action='store_true',
                        help='Force re-extract even if LMDB exists')
    parser.add_argument('--prior_lmdb_map_size_gb', type=float, default=150,
                        help='LMDB map size in GB')
    
    # Sampling
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate samples every N epochs')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    
    # Evaluation (for wandb update)
    parser.add_argument('--max_gen_metrics_samples', type=int, default=5000,
                        help='Max samples for gFID, IS, KID when updating wandb run')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases (and update run if wandb_id)')
    
    return parser.parse_args()


def download_checkpoint_from_wandb(wandb_id: str, entity: str, project: str, download_dir: str) -> str:
    """Download VQ-VAE2 checkpoint from a wandb run."""
    api = wandb.Api()
    run_path = f"{entity}/{project}/{wandb_id}"
    run = api.run(run_path)
    
    def find_pth_in_dir(d):
        for root, _, files in os.walk(d):
            for name in files:
                if name.endswith(".pth"):
                    return os.path.join(root, name)
        return None
    
    pth_files = [f for f in run.files() if fnmatch.fnmatch(f.name, "*.pth") and "prior" not in f.name.lower()]
    pth_files.sort(key=lambda f: (0 if "final_checkpoint" in f.name else 1, f.name))
    for f in pth_files:
        f.download(root=download_dir, replace=True)
        found = find_pth_in_dir(download_dir)
        if found:
            return found
    
    for art in run.logged_artifacts():
        if "model" in art.name.lower() or "final" in art.name.lower():
            art.download(root=download_dir)
            found = find_pth_in_dir(download_dir)
            if found:
                return found
    
    raise FileNotFoundError(
        f"No checkpoint (.pth) found in wandb run {run_path}. "
        "Ensure the run has final_checkpoint.pth (via wandb.save or artifact)."
    )


def load_vqvae2(checkpoint_path, device):
    """Load pre-trained VQ-VAE2 or GG-VQ-VAE2 from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        ckpt_args = checkpoint.get('args', None)
    else:
        model_state = checkpoint
        ckpt_args = None
    
    if 'vq_top.embedding.weight' not in model_state and 'quantize_t.embedding.weight' not in model_state:
        raise ValueError("Cannot infer VQ-VAE2 parameters from checkpoint")
    
    emb_key = 'vq_top.embedding.weight' if 'vq_top.embedding.weight' in model_state else 'quantize_t.embedding.weight'
    num_embeddings = model_state[emb_key].shape[0]
    embedding_dim = model_state[emb_key].shape[1]
    
    if ckpt_args is not None:
        args = SimpleNamespace(**ckpt_args) if isinstance(ckpt_args, dict) else ckpt_args
        input_size = getattr(args, 'input_size', 32)
        if not hasattr(args, 'input_size'):
            args.input_size = input_size
        dataset = getattr(args, 'dataset', 'cifar10')
        args.dataset_size = 50000 if 'cifar' in str(dataset).lower() else 50000
        args.batch_size = 128
        try:
            model = get_network(
                input_size=input_size,
                num_channels=3,
                args=args,
                device=device,
            )
        except Exception as e:
            print(f"get_network failed ({e}), falling back to VQVAE2")
            args = None
    else:
        args = None
    
    if args is None:
        input_size = 32
        model = VQVAE2(
            in_channels=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=[128, 256],
            input_size=input_size,
            device=device,
        )
    
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print(f"Loaded VQ-VAE2: {num_embeddings} embeddings, dim={embedding_dim}")
    print(f"Top latent: {model.latent_spatial_dim_top}x{model.latent_spatial_dim_top}, "
          f"Bottom: {model.latent_spatial_dim_bottom}x{model.latent_spatial_dim_bottom}")
    
    return model, ckpt_args


def train_epoch_images(prior, vqvae2, dataloader, optimizer, device, epoch):
    """Train one epoch extracting codes on-the-fly from images."""
    prior.train()
    loss_top_meter = AverageMeter()
    loss_bottom_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for images, _ in pbar:
        images = images.to(device)
        with torch.no_grad():
            code_dict = vqvae2.get_code_indices(images)
            z_top = code_dict['indices_top']
            z_bottom = code_dict['indices_bottom']
        
        optimizer.zero_grad()
        loss_dict = prior.loss_function(z_top, z_bottom)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optimizer.step()
        
        loss_top_meter.update(loss_dict['loss_top'].item(), images.size(0))
        loss_bottom_meter.update(loss_dict['loss_bottom'].item(), images.size(0))
        loss_total_meter.update(total_loss.item(), images.size(0))
        pbar.set_postfix({'total': f'{loss_total_meter.avg:.4f}'})
    
    return {
        'loss_top': loss_top_meter.avg,
        'loss_bottom': loss_bottom_meter.avg,
        'total_loss': loss_total_meter.avg,
    }


def train_epoch_codes(prior, dataloader, optimizer, device, epoch):
    """Train one epoch from pre-extracted LMDB codes (z_top, z_bottom)."""
    prior.train()
    loss_total_meter = AverageMeter()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        z_top, z_bottom = batch
        z_top, z_bottom = z_top.to(device), z_bottom.to(device)
        
        optimizer.zero_grad()
        loss_dict = prior.loss_function(z_top, z_bottom)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optimizer.step()
        
        loss_total_meter.update(total_loss.item(), z_top.size(0))
        pbar.set_postfix({'total': f'{loss_total_meter.avg:.4f}'})
    
    return {
        'loss_top': loss_total_meter.avg,
        'loss_bottom': loss_total_meter.avg,
        'total_loss': loss_total_meter.avg,
    }


@torch.no_grad()
def generate_samples(prior, vqvae2, num_samples, device, temperature=1.0):
    """Generate samples using prior + VQ-VAE2 decoder."""
    prior.eval()
    vqvae2.eval()
    return prior.sample_with_vqvae2(
        vqvae2_model=vqvae2,
        batch_size=num_samples,
        device=device,
        temperature=temperature,
    )


def main():
    args = parse_args()
    
    if args.vqvae2_checkpoint is None and args.wandb_id is None:
        raise ValueError("Provide either --vqvae2_checkpoint (local path) or --wandb_id (wandb run ID)")
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Resolve checkpoint path
    checkpoint_path = args.vqvae2_checkpoint
    if checkpoint_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Downloading checkpoint from wandb run {args.wandb_entity}/{args.wandb_project}/{args.wandb_id}")
            downloaded = download_checkpoint_from_wandb(
                args.wandb_id, args.wandb_entity, args.wandb_project, tmpdir
            )
            dest = checkpoints_dir / 'downloaded_vqvae2.pth'
            shutil.copy(downloaded, dest)
            checkpoint_path = str(dest)
    
    print(f"Loading VQ-VAE2 from {checkpoint_path}")
    vqvae2, ckpt_args = load_vqvae2(checkpoint_path, device)
    
    normalize = getattr(ckpt_args, 'normalize_inputs', False) if ckpt_args else False
    dataset_name = getattr(ckpt_args, 'dataset', args.dataset) if ckpt_args else args.dataset
    
    train_dataset, test_dataset, input_size = get_dataset(
        dataset_name, args.data_dir, normalize=normalize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    save_root = str(output_dir)
    
    # LMDB extraction (use checkpoint arch for GG-VQ-VAE2)
    arch = getattr(ckpt_args, 'arch', 'vq_vae2') if ckpt_args else 'vq_vae2'
    prior_args = SimpleNamespace(
        arch=arch,
        dataset=dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prior_use_lmdb_codes=args.prior_use_lmdb_codes,
        prior_force_extract_codes=args.prior_force_extract_codes,
        prior_lmdb_map_size_gb=args.prior_lmdb_map_size_gb,
        input_size=input_size,
    )
    
    codes_dataset, use_lmdb = get_or_extract_codes_lmdb(
        vqvae2, train_loader, device, save_root,
        is_hierarchical=True,
        args=prior_args,
        force_extract=args.prior_force_extract_codes,
        map_size=int(args.prior_lmdb_map_size_gb * 1024**3),
    )
    
    if use_lmdb and codes_dataset is not None:
        codes_loader = DataLoader(
            codes_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_data = codes_loader
        use_codes = True
    else:
        train_data = train_loader
        use_codes = False
    
    prior = HierarchicalPixelCNN(
        num_embeddings=vqvae2.num_embeddings,
        embedding_dim=vqvae2.embedding_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    ).to(device)
    
    optimizer = optim.Adam(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    if args.use_wandb and args.wandb_id is None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        if use_codes:
            metrics = train_epoch_codes(prior, train_data, optimizer, device, epoch)
        else:
            metrics = train_epoch_images(prior, vqvae2, train_data, optimizer, device, epoch)
        
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Top: {metrics['loss_top']:.4f}, Bottom: {metrics['loss_bottom']:.4f}, "
              f"Total: {metrics['total_loss']:.4f}")
        
        if args.use_wandb and args.wandb_id is None:
            wandb.log({
                'epoch': epoch,
                'train/loss_top': metrics['loss_top'],
                'train/loss_bottom': metrics['loss_bottom'],
                'train/total_loss': metrics['total_loss'],
            })
        
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            torch.save({
                'epoch': epoch, 'model_state_dict': prior.state_dict(),
                'loss': best_loss, 'args': vars(args),
            }, checkpoints_dir / 'best_prior.pth')
        
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            samples = generate_samples(prior, vqvae2, args.num_samples, device, args.temperature)
            sample_path = samples_dir / f'samples_epoch_{epoch:04d}.png'
            use_tanh = hasattr(vqvae2, 'recons_activation') and vqvae2.recons_activation is not None
            if use_tanh and type(vqvae2.recons_activation).__name__ == 'Tanh':
                save_image(samples, sample_path, nrow=8, normalize=True, value_range=(-1, 1))
            else:
                save_image(samples, sample_path, nrow=8, normalize=True)
            
            if args.use_wandb and args.wandb_id is None:
                wandb.log({'samples': wandb.Image(str(sample_path))})
    
    final_path = checkpoints_dir / 'final_prior.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': prior.state_dict(),
        'loss': metrics['total_loss'],
        'args': vars(args),
    }, final_path)
    
    print(f"Training complete. Best loss: {best_loss:.4f}. Saved to {final_path}")
    
    # Final samples for wandb upload
    num_vis = min(64, args.num_samples)
    final_samples = generate_samples(prior, vqvae2, num_vis, device, args.temperature)
    random_samples_path = samples_dir / 'random_prior_generated_samples.png'
    use_tanh = hasattr(vqvae2, 'recons_activation') and vqvae2.recons_activation is not None
    if use_tanh and type(vqvae2.recons_activation).__name__ == 'Tanh':
        save_image(final_samples, random_samples_path, nrow=8, normalize=True, value_range=(-1, 1))
    else:
        save_image(final_samples, random_samples_path, nrow=8, normalize=True)
    
    # Update wandb run if wandb_id
    if args.use_wandb and args.wandb_id:
        from main import evaluate_generative_metrics
        
        eval_args = SimpleNamespace(
            max_gen_metrics_samples=args.max_gen_metrics_samples,
            batch_size=args.batch_size,
        )
        gen_metrics = evaluate_generative_metrics(
            vqvae2, test_loader, device, eval_args, prior=prior
        )
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_id,
            resume='must',
        )
        
        run.summary.update({
            'final/gfid': gen_metrics['gfid'],
            'final/inception_score_mean': gen_metrics['inception_score_mean'],
            'final/inception_score_std': gen_metrics['inception_score_std'],
            'final/kid': gen_metrics['kid'],
        })
        
        wandb.save(str(final_path), base_path=str(output_dir))
        run.log({'random_prior_generated_samples': wandb.Image(str(random_samples_path))})
        
        wandb.finish()
        print(f"Updated wandb run {args.wandb_entity}/{args.wandb_project}/{args.wandb_id}")
    
    elif args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
