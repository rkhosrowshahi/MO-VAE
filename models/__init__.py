from .vae import VAE
from .vq_vae import VQVAE

def get_network(input_size, num_channels=3, args=None):
    arch = args.arch
    latent_dim = args.latent_dim
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    hidden_dims = args.hidden_dims
    objs = args.objs
    beta = args.beta

    if arch.lower() == 'vae':
        return VAE(latent_dim=latent_dim, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, objs=objs, beta=beta)
    elif arch.lower() == 'vq_vae':
        return VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, input_size=input_size, in_channels=num_channels, objs=objs, beta=beta)
    else:
        raise ValueError(f"Network architecture {arch} not supported")
