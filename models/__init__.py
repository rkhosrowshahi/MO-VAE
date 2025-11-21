from .vae import VAE
from .gg_vae import GGVAE
from .vq_vae import VQVAE
from .vq_vae2 import VQVAE2
from .betatc_vae import BetaTCVAE

def get_network(input_size, num_channels=3, args=None, device=None):
    arch = getattr(args, "arch", "vae")
    latent_dim = getattr(args, "latent_dim", 128)
    embedding_dim = getattr(args, "embedding_dim", 64)
    num_embeddings = getattr(args, "num_embeddings", 512)
    hidden_dims = getattr(args, "hidden_dims", [32, 64, 128, 256, 512])
    recons_dist = getattr(args, "recons_dist", "gaussian")
    recons_reduction = getattr(args, "recons_reduction", "mean")
    kld_weight = getattr(args, "kld_weight", 0.00025)
    beta = getattr(args, "beta", 1.0)
    alpha = getattr(args, "alpha", 1.0)
    gamma = getattr(args, "gamma", 1.0)
    # For BetaTCVAE, gamma defaults to 1.0 (for KLD annealing)
    # Note: args.gamma is also used for scheduler, but they're used in different contexts
    anneal_steps = getattr(args, 'anneal_steps', 200)
    dataset_size = getattr(args, 'dataset_size', 50000)
    output_activation = getattr(args, 'output_activation', 'tanh')

    if arch.lower() == 'vae':
        return VAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, kld_weight=kld_weight, beta=beta, device=device)
    elif arch.lower() == 'gg_vae':
        return GGVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, kld_weight=kld_weight, beta=beta, device=device)
    elif arch.lower() == 'vq_vae':
        return VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, beta=beta, device=device)
    elif arch.lower() == 'vq_vae2':
        return VQVAE2(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, recons_dist=recons_dist, recons_reduction=recons_reduction, beta=beta, device=device)
    elif arch.lower() == 'betatc_vae' or arch.lower() == 'btc_vae':
        return BetaTCVAE(
            in_channels=num_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            anneal_steps=anneal_steps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            input_size=input_size,
            dataset_size=dataset_size,
            output_activation=output_activation,
            recons_dist=recons_dist,
            recons_reduction=recons_reduction,
            device=device
        )
    else:
        raise ValueError(f"Network architecture {arch} not supported")
