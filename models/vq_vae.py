from typing import List, Optional, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

from utils.objectives import mse_recon_sum, mse_recon_mean

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()

        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> tuple:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape

        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), commitment_loss, embedding_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):
    """
    Residual layer for VQ-VAE encoder and decoder
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels,
                     kernel_size=1, bias=False)
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) model
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 beta: float = 0.25,
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 objs: Optional[List[str]] = ["mse_sum"],
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.input_size = input_size
        self.in_channels = in_channels
        self.beta = beta
        
        # Calculate spatial dimensions of latent space
        # Each encoder layer with stride=2 halves the spatial dimension
        num_downsamples = len(hidden_dims)
        self.latent_spatial_dim = input_size // (2 ** num_downsamples)
        

        recon_obj = None
        if "mse_sum" in objs:
            recon_obj = mse_recon_sum
        elif "mse_mean" in objs:
            recon_obj = mse_recon_mean
        else:
            raise ValueError(f"Reconstruction objective {objs} not supported")
        self.recon_obj = recon_obj

        self.objectives = {"reconstruction_loss": recon_obj, "commitment_loss": None, "embedding_loss": None}

        modules = []
        
        # Store original hidden_dims for decoder
        encoder_hidden_dims = hidden_dims.copy()
        
        # Setup layer normalization
        if layer_norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif layer_norm == "layer":
            norm_layer = nn.LayerNorm
        elif layer_norm == "none":
            norm_layer = lambda x: nn.Identity()
        else:
            raise ValueError(f"Layer norm {layer_norm} not supported")
        
        # Setup output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Output activation {output_activation} not supported")

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))

        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          encoder_hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(encoder_hidden_dims[-1], encoder_hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        decoder_hidden_dims = encoder_hidden_dims.copy()
        decoder_hidden_dims.reverse()

        for i in range(len(decoder_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_hidden_dims[i],
                                       decoder_hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_hidden_dims[-1],
                                   out_channels=self.in_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                self.output_activation))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.

        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> Dict[str, Any]:
        encoding = self.encode(input)
        quantized_inputs, commitment_loss, embedding_loss = self.vq_layer(encoding)
        return {"recons": self.decode(quantized_inputs), "quantized_inputs": quantized_inputs, "commitment_loss": commitment_loss, "embedding_loss": embedding_loss}
    
    def loss_function(self, inputs, args: dict) -> dict:
        """
        Computes the VQ-VAE loss function.
        
        :param args: Dictionary containing the arguments
        :return: Dictionary containing the losses
        """

        recons = args["recons"]
        commitment_loss = args["commitment_loss"]
        embedding_loss = self.beta * args["embedding_loss"]
        recon_loss = self.recon_obj(recons, inputs)
        
        return {
            "reconstruction_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
        }

    def sample(self, num_samples=1, device=None):
        """
        Sample from the latent space and generate images.
        
        For VQ-VAE, we randomly select indices from the codebook and 
        decode the corresponding embeddings.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images of shape [num_samples, in_channels, input_size, input_size]
        """
        self.eval()
        with torch.no_grad():
            # Randomly sample indices from the codebook
            # Shape: [num_samples, latent_spatial_dim, latent_spatial_dim]
            random_indices = torch.randint(
                0, self.num_embeddings,
                (num_samples, self.latent_spatial_dim, self.latent_spatial_dim),
                device=device
            )
            
            # Get embeddings from the codebook
            # Flatten indices: [num_samples * H * W]
            flat_indices = random_indices.view(-1)
            
            # Get embeddings: [num_samples * H * W, embedding_dim]
            quantized = self.vq_layer.embedding(flat_indices)
            
            # Reshape to [num_samples, H, W, embedding_dim]
            quantized = quantized.view(
                num_samples, 
                self.latent_spatial_dim, 
                self.latent_spatial_dim, 
                self.embedding_dim
            )
            
            # Permute to [num_samples, embedding_dim, H, W] for decoder
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
            # Decode to generate images
            generated_samples = self.decode(quantized)
            
        return generated_samples

    def total_trainable_params(self):
        """
        Returns the total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """
        Prints the model summary
        """
        return summary(self, (self.in_channels, self.input_size, self.input_size))
