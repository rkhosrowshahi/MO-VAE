from .vae import VAE
from .gg_vae import GGVAE
from .recursive_kl_vae import RecursiveKLVAE
from .cycle_vae import CycleVAE
from .recursive_cyclic_vae import RecursiveCyclicVAE
from .sphere_encoder import SphereEncoder
from .sphere_encoder_vit import SphereEncoderViT
from .vq_vae import VQVAE
from .gg_vq_vae import GGVQVAE
from .vq_vae2 import VQVAE2
from .betatc_vae import BetaTCVAE
from .pixelcnn_prior import PixelCNN, HierarchicalPixelCNN

def get_network(input_size, num_channels=3, args=None, device=None):
    arch = getattr(args, "arch", "vae")
    latent_dim = getattr(args, "latent_dim", 128)
    embedding_dim = getattr(args, "embedding_dim", 64)
    num_embeddings = getattr(args, "num_embeddings", 512)
    hidden_dims = getattr(args, "hidden_dims", [32, 64, 128, 256, 512])
    num_residual_layers = getattr(args, "num_residual_layers", 2) # For VQVAE
    recons_dist = getattr(args, "recons_dist", "gaussian")
    recons_reduction = getattr(args, "recons_reduction", "mean")
    # Support both loss_weights (new) and lambda_weights (old) for backward compatibility
    lambda_weights = getattr(args, "loss_weights", None) or getattr(args, "lambda_weights", None)
    anneal_steps = getattr(args, 'anneal_steps', 200)
    dataset_size = getattr(args, 'dataset_size', 50000)

    if arch.lower() == 'vae':
        # Default lambda_weights for VAE: [reconstruction, kld]
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size]
        else:
            lambda_weights[1] = args.batch_size / args.dataset_size
        return VAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'recursive_kl_vae':
        # Recursive KL: [reconstruction, recursive_kld]
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size]
        recursive_kld_anneal_steps = getattr(args, 'recursive_kld_anneal_steps', 25000)
        return RecursiveKLVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, recursive_kld_anneal_steps=recursive_kld_anneal_steps, device=device)
    elif arch.lower() == 'cycle_vae':
        # Cycle VAE: [reconstruction, cycle] (two weights only)
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size]
        return CycleVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'recursive_cyclic_vae' or arch.lower() == 'rc_vae':
        # Recursive Cyclic VAE: [reconstruction, recursive_kld, cycle]
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size, args.batch_size / args.dataset_size]
        recursive_kld_anneal_steps = getattr(args, 'recursive_kld_anneal_steps', 25000)
        return RecursiveCyclicVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, recursive_kld_anneal_steps=recursive_kld_anneal_steps, device=device)
    elif arch.lower() == 'sphere_encoder':
        # Sphere Encoder (arXiv:2602.15030): spherical latent, pix_recon + pix_con + lat_con
        # lambda_weights unused; use model kwargs for sigma_max_angle_deg, lambda_* if needed
        sigma_max_angle_deg = getattr(args, "sigma_max_angle_deg", 80.0)
        sigma_mix_prob = getattr(args, "sigma_mix_prob", 0.0)
        sigma_mix_angle_min_deg = getattr(args, "sigma_mix_angle_min_deg", None)
        sigma_mix_angle_max_deg = getattr(args, "sigma_mix_angle_max_deg", None)
        lambda_pix_recon = getattr(args, "lambda_pix_recon", 1.0)
        lambda_pix_con = getattr(args, "lambda_pix_con", 0.5)
        lambda_lat_con = getattr(args, "lambda_lat_con", 0.1)
        return SphereEncoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            input_size=input_size,
            in_channels=num_channels,
            recons_dist=recons_dist,
            recons_reduction=recons_reduction,
            lambda_weights=[1.0, 0.0],
            sigma_max_angle_deg=sigma_max_angle_deg,
            sigma_mix_prob=sigma_mix_prob,
            sigma_mix_angle_min_deg=sigma_mix_angle_min_deg,
            sigma_mix_angle_max_deg=sigma_mix_angle_max_deg,
            lambda_pix_recon=lambda_pix_recon,
            lambda_pix_con=lambda_pix_con,
            lambda_lat_con=lambda_lat_con,
            device=device,
        )
    elif arch.lower() == 'sphere_encoder_vit':
        # Sphere Encoder ViT (paper arch): ViT + MLP-Mixer + RoPE + sinusoidal
        sigma_max_angle_deg = getattr(args, "sigma_max_angle_deg", 80.0)
        sigma_mix_prob = getattr(args, "sigma_mix_prob", 0.0)
        sigma_mix_angle_min_deg = getattr(args, "sigma_mix_angle_min_deg", None)
        sigma_mix_angle_max_deg = getattr(args, "sigma_mix_angle_max_deg", None)
        lambda_pix_recon = getattr(args, "lambda_pix_recon", 1.0)
        lambda_pix_con = getattr(args, "lambda_pix_con", 0.5)
        lambda_lat_con = getattr(args, "lambda_lat_con", 0.1)
        patch_size = getattr(args, "patch_size", 2 if input_size <= 32 else 8)
        num_patches = (input_size // patch_size) ** 2
        # latent_dim = L (total spherical dim); latent_channels = L // num_patches per token
        L = latent_dim
        latent_channels = L // num_patches
        if L != latent_channels * num_patches:
            raise ValueError(f"sphere_encoder_vit: latent_dim {L} must be divisible by num_patches {num_patches}")
        embed_dim = getattr(args, "vit_embed_dim", 1024)
        depth = getattr(args, "vit_depth", 24)
        num_heads = getattr(args, "vit_num_heads", 16)
        mixer_depth = getattr(args, "vit_mixer_depth", 2)
        return SphereEncoderViT(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=num_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            mixer_depth=mixer_depth,
            latent_channels=latent_channels,
            num_classes=getattr(args, "num_classes", 0),
            sigma_max_angle_deg=sigma_max_angle_deg,
            sigma_mix_prob=sigma_mix_prob,
            sigma_mix_angle_min_deg=sigma_mix_angle_min_deg,
            sigma_mix_angle_max_deg=sigma_mix_angle_max_deg,
            lambda_pix_recon=lambda_pix_recon,
            lambda_pix_con=lambda_pix_con,
            lambda_lat_con=lambda_lat_con,
            device=device,
        )
    elif arch.lower() == 'gg_vae':
        # Default lambda_weights for GGVAE: [reconstruction, kld, gradient_guided, edge_matching]
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size, 1.0, 1.0]
        else:
            lambda_weights[1] = args.batch_size / args.dataset_size
        return GGVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, edge_matching_version=1)
    elif arch.lower() in ('gg_vae_v2', 'gg_vae_v3', 'gg_vae_v4', 'gg_vae_v5', 'gg_vae_v6'):
        # GGVAE with different edge_matching_loss versions (v2=norm L1, v3=angle, v4=masked, v5=cosine, v6=binary BCE)
        version = int(arch.lower().split('_')[-1].replace('v', ''))
        if lambda_weights is None:
            lambda_weights = [1.0, args.batch_size / args.dataset_size, 1.0, 1.0]
        else:
            lambda_weights[1] = args.batch_size / args.dataset_size
        return GGVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, edge_matching_version=version)
    elif arch.lower() == 'vq_vae':
        # Default lambda_weights for VQVAE: [reconstruction, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.25]
        return VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, num_residual_layers=num_residual_layers, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'gg_vq_vae_v1' or arch.lower() == 'gg_vq_vae':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.25, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, num_residual_layers=num_residual_layers, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v1")
    elif arch.lower() == 'gg_vq_vae_v2':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.25, 1.0, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, num_residual_layers=num_residual_layers, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v2")
    elif arch.lower() == 'gg_vq_vae_v3':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.25, 1.0, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, num_residual_layers=num_residual_layers, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v3")
    elif arch.lower() == 'vq_vae2':
        # Default lambda_weights for VQVAE2: [reconstruction, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.25]
        return VQVAE2(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, num_residual_layers=num_residual_layers, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'betatc_vae' or arch.lower() == 'btc_vae':
        # Default lambda_weights for BetaTCVAE: [reconstruction, mi, tc, kld]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0, args.batch_size / args.dataset_size]
        else:
            lambda_weights[3] = args.batch_size / args.dataset_size
        return BetaTCVAE(
            in_channels=num_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            anneal_steps=anneal_steps,
            input_size=input_size,
            dataset_size=dataset_size,
            recons_dist=recons_dist,
            recons_reduction=recons_reduction,
            lambda_weights=lambda_weights,
            device=device
        )
    else:
        raise ValueError(f"Network architecture {arch} not supported")
