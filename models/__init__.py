from .vae import VAE
from .gg_vae import GGVAE
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
    recons_dist = getattr(args, "recons_dist", "gaussian")
    recons_reduction = getattr(args, "recons_reduction", "mean")
    # Support both loss_weights (new) and lambda_weights (old) for backward compatibility
    lambda_weights = getattr(args, "loss_weights", None) or getattr(args, "lambda_weights", None)
    anneal_steps = getattr(args, 'anneal_steps', 200)
    dataset_size = getattr(args, 'dataset_size', 50000)

    if arch.lower() == 'vae':
        # Default lambda_weights for VAE: [reconstruction, kld]
        if lambda_weights is None:
            lambda_weights = [1.0, 0.00025]
        return VAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'gg_vae':
        # Default lambda_weights for GGVAE: [reconstruction, gradient_guided, kld]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 0.00025]
        return GGVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'vq_vae':
        # Default lambda_weights for VQVAE: [reconstruction, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0]
        return VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'gg_vq_vae_v1' or arch.lower() == 'gg_vq_vae':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v1")
    elif arch.lower() == 'gg_vq_vae_v2':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v2")
    elif arch.lower() == 'gg_vq_vae_v3':
        # Default lambda_weights for GGVQVAE: [reconstruction, gradient_guided, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        return GGVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device, version="v3")
    elif arch.lower() == 'vq_vae2':
        # Default lambda_weights for VQVAE2: [reconstruction, commitment, embedding]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0]
        return VQVAE2(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, lambda_weights=lambda_weights, device=device)
    elif arch.lower() == 'betatc_vae' or arch.lower() == 'btc_vae':
        # Default lambda_weights for BetaTCVAE: [reconstruction, mi, tc, kld]
        if lambda_weights is None:
            lambda_weights = [1.0, 1.0, 1.0, 1.0]
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
