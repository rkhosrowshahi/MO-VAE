from typing import List, Optional, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

from utils.objectives import bce_per_pixel_mean, bce_per_image_sum, mse_per_image_sum, mse_per_pixel_mean, mse_total_batch_sum_scaled
from utils.objectives import laplacian_per_image_sum, laplacian_per_pixel_mean

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int):
        super(VectorQuantizer, self).__init__()

        self.K = num_embeddings
        self.D = embedding_dim
        self._summary_mode = False

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

        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()  # [B x D x H x W]

        if getattr(self, "_summary_mode", False):
            return quantized_latents

        return quantized_latents, commitment_loss, embedding_loss


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


class GGVQVAE(nn.Module):
    """
    Gradient-Guided Vector Quantized Variational Autoencoder (GGVQ-VAE) model
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 recons_dist: str = "gaussian",
                 recons_reduction: str = "mean",
                 lambda_weights: Optional[List[float]] = None,
                 device=None,
                 **kwargs) -> None:
        super(GGVQVAE, self).__init__()

        self.device = device
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.input_size = input_size
        self.in_channels = in_channels
        self._summary_mode = False
        
        # Calculate spatial dimensions of latent space
        # Each encoder layer with stride=2 halves the spatial dimension
        num_downsamples = len(hidden_dims)
        self.latent_spatial_dim = input_size // (2 ** num_downsamples)
        

        recon_obj = None
        if recons_dist == "gaussian":
            if recons_reduction == "mean":
                recon_obj = mse_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = mse_per_image_sum
            elif recons_reduction == "scaled_sum":
                recon_obj = mse_total_batch_sum_scaled
            else:
                raise ValueError(f"MSE reduction {recons_reduction} not supported. Choose from: mean, sum, scaled_sum")

            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for gaussian
        elif recons_dist == "bernoulli":
            if recons_reduction == "mean":
                recon_obj = bce_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = bce_per_image_sum
            else:
                 raise ValueError(f"BCE reduction {recons_reduction} not supported. Choose from: mean, sum")
            output_activation = "sigmoid"
        elif recons_dist == "laplacian":
            if recons_reduction == "mean":
                recon_obj = laplacian_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = laplacian_per_image_sum
            else:
                 raise ValueError(f"Laplacian reduction {recons_reduction} not supported. Choose from: mean, sum")
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for laplacian
        else:
            raise ValueError(f"Reconstruction distribution {recons_dist} not supported. Choose from: gaussian, bernoulli, laplacian")
        self.recon_obj = recon_obj

        sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0.,  2.],
                            [-1,  0.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)
        # Ensuring sobel filters can apply to RGB images - register as buffers so they move with model
        self.register_buffer('sobel_x', sobel_x.expand(3, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.expand(3, 1, 3, 3))

        self.objectives = {"reconstruction_loss": recon_obj, 'gradient_guided_loss': self.gradient_guided_loss, "commitment_loss": None, "embedding_loss": None}

        self.features = ["encoding"]

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            lambda_weights = {"reconstruction_loss": 1.0, "gradient_guided_loss": 1.0, "commitment_loss": 1.0, "embedding_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict: [reconstruction_weight, gradient_guided_weight, commitment_weight, embedding_weight]
            if len(lambda_weights) != 4:
                raise ValueError(f"GGVQVAE requires 4 lambda_weights (reconstruction, gradient_guided, commitment, embedding), got {len(lambda_weights)}")
            lambda_weights = {
                "reconstruction_loss": lambda_weights[0],
                "gradient_guided_loss": lambda_weights[1],
                "commitment_loss": lambda_weights[2],
                "embedding_loss": lambda_weights[3]
            }
        elif isinstance(lambda_weights, dict):
            # Validate dict keys match objectives
            expected_keys = set(self.objectives.keys())
            provided_keys = set(lambda_weights.keys())
            if expected_keys != provided_keys:
                missing = expected_keys - provided_keys
                extra = provided_keys - expected_keys
                error_msg = f"lambda_weights keys must match objectives keys. "
                if missing:
                    error_msg += f"Missing: {missing}. "
                if extra:
                    error_msg += f"Extra: {extra}."
                raise ValueError(error_msg)
        else:
            raise TypeError(f"lambda_weights must be dict or list, got {type(lambda_weights)}")
        
        self.lambda_weights = lambda_weights

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
                                        embedding_dim)

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
        vq_outputs = self.vq_layer(encoding)

        if isinstance(vq_outputs, tuple):
            quantized_inputs, commitment_loss, embedding_loss = vq_outputs
        else:
            quantized_inputs = vq_outputs
            commitment_loss, embedding_loss = None, None

        outputs = {
            "recons": self.decode(quantized_inputs),
            "quantized_inputs": quantized_inputs,
            "encoding": encoding,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
        }
        if getattr(self, "_summary_mode", False):
            return outputs["recons"]
        return outputs

    def gradient_guided_loss(self,inputs, recons):
        # Sobel applied to R,G,B
        x_grad = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        y_grad = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))

        #Gradient magnitude
        grad_mag = torch.sqrt(x_grad**2 + y_grad**2)# (batch_size,C,H,W)

        # Combining across channels - take max across channels
        grad_max = torch.max(grad_mag, dim=1)[0]

        # Normalizing
        grad_max = (grad_max - grad_max.view(grad_max.size(0), -1).min(1, keepdim=True)[0].unsqueeze(-1)) / \
                            (grad_max.view(grad_max.size(0), -1).max(1, keepdim=True)[0].unsqueeze(-1) -
                            grad_max.view(grad_max.size(0), -1).min(1, keepdim=True)[0].unsqueeze(-1))

        # non-reduced reconstruction loss (B, C, H, W)
        pixel_loss = F.mse_loss(recons, inputs, reduction='none')

        #Gradient-guided Encoder Loss
        loss_grad = (grad_max.unsqueeze(1) * pixel_loss).mean() #(batch_size, C, H, W) then mean across batch

        return loss_grad
    
    def loss_function(self, inputs, args: dict) -> dict:
        """
        Computes the VQ-VAE loss function.
        
        :param args: Dictionary containing the arguments
        :return: Dictionary containing the losses
        """

        recons = args["recons"]
        commitment_loss = args["commitment_loss"]
        embedding_loss = args["embedding_loss"]
        recon_loss = self.recon_obj(inputs, recons)
        gradient_guided_loss = self.gradient_guided_loss(inputs, recons)
        
        # Apply lambda_weights using dictionary keys matching self.objectives
        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_gradient_guided_loss = self.lambda_weights["gradient_guided_loss"] * gradient_guided_loss
        weighted_commitment_loss = self.lambda_weights["commitment_loss"] * commitment_loss
        weighted_embedding_loss = self.lambda_weights["embedding_loss"] * embedding_loss

        return {
            "reconstruction_loss": weighted_recon_loss,
            "gradient_guided_loss": weighted_gradient_guided_loss,
            "commitment_loss": weighted_commitment_loss,
            "embedding_loss": weighted_embedding_loss,
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
        was_training = self.training
        try:
            self._summary_mode = True
            self.vq_layer._summary_mode = True
            self.train(False)
            param_device = next(self.parameters()).device
            summary_device = "cuda" if param_device.type == "cuda" else "cpu"
            result = summary(
                self,
                (self.in_channels, self.input_size, self.input_size),
                device=summary_device,
            )
            return result
        except Exception as e:
            print(f"Error printing model summary: {e}")
            return None
        finally:
            self._summary_mode = False
            self.vq_layer._summary_mode = False
            self.train(was_training)
