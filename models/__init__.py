from .vae import VAE
from .vq_vae import VQVAE

def get_network(input_size, num_channels=3, args=None):
    arch = getattr(args, "arch", "vae")
    latent_dim = getattr(args, "latent_dim", 128)
    embedding_dim = getattr(args, "embedding_dim", 64)
    num_embeddings = getattr(args, "num_embeddings", 512)
    hidden_dims = getattr(args, "hidden_dims", [32, 64, 128, 256, 512])
    objs = args.objs
    kld_weight = getattr(args, "kld_weight", 0.00025)
    beta = getattr(args, "beta", 1.0)

    if arch.lower() == 'vae':
        return VAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, objs=objs, kld_weight=kld_weight, beta=beta)
    elif arch.lower() == 'vq_vae':
        return VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, objs=objs, beta=beta)
    else:
        raise ValueError(f"Network architecture {arch} not supported")
