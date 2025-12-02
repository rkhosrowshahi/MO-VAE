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

        return quantized_latents, embedding_loss, commitment_loss


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
                 version: str = "v1",
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

        self.objectives = {"reconstruction_loss": recon_obj, "embedding_loss": None, "commitment_loss": None}
        
        if version == "v1":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
        elif version == "v2":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
            self.objectives["edge_matching_loss"] = self.edge_matching_loss_v1

        elif version == "v3":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
            self.objectives["edge_matching_loss"] = self.edge_matching_loss_v2
        else:
            raise ValueError(f"Version {version} not supported. Choose from: v1, v2, v3")

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            # Default weights based on version
            if version == "v1":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0}
            elif version == "v2":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
            elif version == "v3":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict based on version
            if version == "v1":
                if len(lambda_weights) != 4:
                    raise ValueError(f"GGVQVAE v1 requires 4 lambda_weights (reconstruction, embedding, commitment, gradient_guided), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3]
                }
            elif version == "v2":
                if len(lambda_weights) != 5:
                    raise ValueError(f"GGVQVAE v2 requires 5 lambda_weights (reconstruction, embedding, commitment, gradient_guided, edge_matching), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3],
                    "edge_matching_loss": lambda_weights[4]
                }
            elif version == "v3":
                if len(lambda_weights) != 5:
                    raise ValueError(f"GGVQVAE v3 requires 5 lambda_weights (reconstruction, embedding, commitment, gradient_guided, edge_matching), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3],
                    "edge_matching_loss": lambda_weights[4]
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

        self.features = ["encoding"]

        modules = []
        
        # Store original hidden_dims for decoder
        encoder_hidden_dims = hidden_dims.copy()
        
        # Setup output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
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
            quantized_inputs, embedding_loss, commitment_loss = vq_outputs
        else:
            quantized_inputs = vq_outputs
            embedding_loss, commitment_loss = None, None

        outputs = {
            "recons": self.decode(quantized_inputs),
            "quantized_inputs": quantized_inputs,
            "encoding": encoding,
            "embedding_loss": embedding_loss,
            "commitment_loss": commitment_loss,
        }
        if getattr(self, "_summary_mode", False):
            return outputs["recons"]
        return outputs

    def gradient_guided_loss(self, inputs, recons):
        # # Sobel applied to R,G,B
        # x_grad = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        # y_grad = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))

        # #Gradient magnitude
        # grad_mag = torch.sqrt(x_grad**2 + y_grad**2 + 1e-8)# (batch_size,C,H,W)

        # # Combining across channels - take max across channels
        # grad_max = torch.max(grad_mag, dim=1)[0]

        # # Normalizing
        # flat = grad_max.view(grad_max.size(0), -1)
        # min_val = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        # max_val = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        # grad_max = (grad_max - min_val) / (max_val - min_val + 1e-8)

        # # non-reduced reconstruction loss (B, C, H, W)
        # pixel_loss = F.mse_loss(recons, inputs, reduction='none')
        # # pixel_loss = F.binary_cross_entropy(recons, inputs, reduction='none')

        # #Gradient-guided Encoder Loss
        # loss_grad = (grad_max.unsqueeze(1) * pixel_loss).mean() #(batch_size, C, H, W) then mean across batch

        # return loss_grad

        eps = 1e-8
        
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge-weighted pixel loss: BCE (pixels are in [0,1])
        grad_target = torch.sqrt(input_x**2 + input_y**2 + eps) #(batch_size,C,H,W)
        weights = grad_target.max(dim=1)[0]  # simplified
        weights = weights / (weights.max() + eps)  # normalize to [0,1]
        
        pixel_loss = F.binary_cross_entropy(recons, inputs, reduction='none')  # BCE here
        weighted_pixel_loss = (weights.unsqueeze(1) * pixel_loss).mean()
        
        return weighted_pixel_loss

    def edge_matching_loss_v1(self, inputs, recons):
        eps = 1e-8
        
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge matching: MSE (gradients are signed, unbounded)
        edge_match_loss = F.mse_loss(recon_x, input_x) + F.mse_loss(recon_y, input_y)
        
        return edge_match_loss

    def edge_matching_loss_v2(self, inputs, recons):
        eps = 1e-8
        
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge matching: L1 loss (gradients are signed, unbounded)
        # Computes gradient magnitudes and compares them with L1
        # Purpose: Forces the model to reproduce edge structures
        grad_pred = torch.sqrt(recon_x**2 + recon_y**2 + eps)
        grad_target = torch.sqrt(input_x**2 + input_y**2 + eps)

        edge_match_loss = F.l1_loss(grad_pred, grad_target)
        
        return edge_match_loss
    
    def loss_function(self, inputs, args: dict) -> dict:
        """
        Computes the VQ-VAE loss function.
        
        :param args: Dictionary containing the arguments
        :return: Dictionary containing the losses
        """

        recons = args["recons"]
        embedding_loss = args["embedding_loss"]
        commitment_loss = args["commitment_loss"]

        loss_dict = {}
        for key, value in self.objectives.items():
            if key == "embedding_loss":
                weighted_loss = embedding_loss
            elif key == "commitment_loss":
                weighted_loss = commitment_loss
            else:
                weighted_loss = value(inputs, recons)
            
            loss_dict[key] = self.lambda_weights[key] * weighted_loss

        return loss_dict

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
        # Get the device the model is actually on (from its parameters)
        model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        # Determine summary device (torchsummary only accepts 'cuda' or 'cpu')
        if model_device.type == 'cuda':
            summary_device = "cuda"
            # If model is on a specific CUDA device (e.g., cuda:7), temporarily set it as default
            # so torchsummary creates inputs on the correct device
            if model_device.index is not None:
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(model_device.index)
        else:
            summary_device = "cpu"
            original_device = None
        
        try:
            self._summary_mode = True
            self.vq_layer._summary_mode = True
            self.train(False)
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
            # Restore original CUDA device if we changed it
            if model_device.type == 'cuda' and model_device.index is not None and original_device is not None:
                torch.cuda.set_device(original_device)
            self._summary_mode = False
            self.vq_layer._summary_mode = False
            self.train(was_training)
