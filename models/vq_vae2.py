
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

from utils.objectives import mse_per_image_sum, mse_per_pixel_mean, mse_total_batch_sum_scaled
from utils.objectives import bce_with_logits_per_image_sum, bce_with_logits_per_pixel_mean
from utils.objectives import laplacian_per_image_sum, laplacian_per_pixel_mean
from models.vq_vae import VectorQuantizer


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, output_activation="none"
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        if output_activation == "tanh":
            blocks.append(nn.Tanh())
        elif output_activation == "sigmoid":
            blocks.append(nn.Sigmoid())
        elif output_activation == "none":
            pass
        else:
            raise ValueError(f"Output activation {output_activation} not supported")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE2(nn.Module):
    """
    Hierarchical Vector Quantized Variational Autoencoder (VQ-VAE-2) model.
    Uses two levels of latent variables (top and bottom).
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 num_residual_layers: int = 2,
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 recons_dist: str = "gaussian",
                 recons_reduction: str = "mean",
                 lambda_weights: Optional[List[float]] = None,
                 device=None,
                 **kwargs) -> None:
        super(VQVAE2, self).__init__()

        self.device = device
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_residual_layers = num_residual_layers
        self.input_size = input_size
        self.in_channels = in_channels
        self._summary_mode = False
        
        # Set up reconstruction objective
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
                recon_obj = bce_with_logits_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = bce_with_logits_per_image_sum
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

        self.objectives = {
            "reconstruction_loss": recon_obj,
            "commitment_loss": None,
            "embedding_loss": None
        }

        self.features = ["encoding_top", "encoding_bottom"]

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            lambda_weights = {"reconstruction_loss": 1.0, "commitment_loss": 1.0, "embedding_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict: [reconstruction_weight, commitment_weight, embedding_weight]
            if len(lambda_weights) != 3:
                raise ValueError(f"VQVAE2 requires 3 lambda_weights (reconstruction, commitment, embedding), got {len(lambda_weights)}")
            lambda_weights = {
                "reconstruction_loss": lambda_weights[0],
                "commitment_loss": lambda_weights[1],
                "embedding_loss": lambda_weights[2]
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

        # Setup output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Output activation {output_activation} not supported")

        self.enc_b = Encoder(in_channels, hidden_dims[0], num_residual_layers, 32, stride=4)
        self.enc_t = Encoder(hidden_dims[0], hidden_dims[0], num_residual_layers, 32, stride=2)
        self.quantize_conv_t = nn.Conv2d(hidden_dims[0], embedding_dim, 1)
        self.quantize_t = VectorQuantizer(embedding_dim, num_embeddings)
        self.dec_t = Decoder(
            embedding_dim, embedding_dim, hidden_dims[0], num_residual_layers, 32, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embedding_dim + hidden_dims[0], embedding_dim, 1)
        self.quantize_b = VectorQuantizer(embedding_dim, num_embeddings)
        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embedding_dim + embedding_dim,
            in_channels,
            hidden_dims[0],
            num_residual_layers,
            32,
            stride=4,
            output_activation=output_activation
        )
        # Latent spatial dims from encoder strides: enc_b stride=4 (÷4), enc_t stride=2 (÷2)
        self.latent_spatial_dim_bottom = input_size // 4   # enc_b
        self.latent_spatial_dim_top = input_size // 8     # enc_b then enc_t

    def encode(self, input: Tensor) -> List[Tensor]:
        # Encode to bottom representation
        enc_b = self.enc_b(input)
        # Encode bottom representation to top representation
        enc_t = self.enc_t(enc_b)
        
        quant_t = self.quantize_conv_t(enc_t)
        quant_t, commitment_loss_t, embedding_loss_t, encoding_inds_top = self.quantize_t(quant_t)

        dec_t = self.dec_t(quant_t)
        dec_t_enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(dec_t_enc_b)
        quant_b, commitment_loss_b, embedding_loss_b, encoding_inds_bottom = self.quantize_b(quant_b)

        return enc_b, enc_t, quant_t, quant_b, commitment_loss_t, commitment_loss_b, embedding_loss_t, embedding_loss_b, encoding_inds_top, encoding_inds_bottom

    def decode(self, quant_t: Tensor, quant_b: Tensor) -> Tensor:
        # Decode top
        dec_t = self.upsample_t(quant_t)
        # Concatenate with bottom quantization
        dec_input = torch.cat([dec_t, quant_b], dim=1)
        # Decode bottom
        recons = self.dec(dec_input)
        return recons

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        recons = self.decode(quant_t, quant_b)

        return recons

    def forward(self, input: Tensor, **kwargs) -> Dict[str, Any]:
        enc_b, enc_t, quant_t, quant_b, commitment_loss_t, commitment_loss_b, embedding_loss_t, embedding_loss_b, encoding_inds_top, encoding_inds_bottom = self.encode(input)

        recons = self.decode(quant_t, quant_b)

        # Sum losses
        commitment_loss = commitment_loss_t + commitment_loss_b
        embedding_loss = embedding_loss_t + embedding_loss_b # In existing implementation this variable holds embedding loss

        # Calculate codebook usage for both codebooks
        codebook_usage_percentage = 0.0
        if encoding_inds_top is not None and encoding_inds_bottom is not None:
            # Combine both codebooks for overall utilization
            # For VQVAE2, we typically report the average or combined utilization
            usage_top = self.quantize_t.get_codebook_usage_percentage_from_indices(encoding_inds_top)
            usage_bottom = self.quantize_b.get_codebook_usage_percentage_from_indices(encoding_inds_bottom)
            codebook_usage_percentage = (usage_top + usage_bottom) / 2.0
        
        outputs = {
            "recons": recons,
            "encoding_top": enc_t,
            "encoding_bottom": enc_b,
            "quantized_top": quant_t,
            "quantized_bottom": quant_b,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
            "codebook_usage_percentage": codebook_usage_percentage,
            "encoding_inds_top": encoding_inds_top,
            "encoding_inds_bottom": encoding_inds_bottom,
        }
        
        if getattr(self, "_summary_mode", False):
            return outputs["recons"]
            
        return outputs

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        commitment_loss = args["commitment_loss"]
        embedding_loss = args["embedding_loss"]
        recon_loss = self.recon_obj(inputs, recons)
        
        # Apply lambda_weights using dictionary keys matching self.objectives
        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_commitment_loss = self.lambda_weights["commitment_loss"] * commitment_loss
        weighted_embedding_loss = self.lambda_weights["embedding_loss"] * embedding_loss
        total_loss = (
            weighted_recon_loss
            + weighted_commitment_loss
            + weighted_embedding_loss
        )

        return {
            "reconstruction_loss": weighted_recon_loss,
            "commitment_loss": weighted_commitment_loss,
            "embedding_loss": weighted_embedding_loss,
            "total_loss": total_loss,
        }

    def sample(self, num_samples=1, device=None):
        """
        Sample uniformly from codebooks (random noise generation).
        Requires a prior (e.g. PixelCNN) for meaningful generation.

        For proper sampling with learned prior, use:
            from models.pixelcnn_prior import HierarchicalPixelCNN
            prior = HierarchicalPixelCNN(...)
            samples = prior.sample_with_vqvae2(vqvae2_model, batch_size, device)
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            # Random top indices [B, H_t, W_t]
            code_t = torch.randint(
                0,
                self.num_embeddings,
                (num_samples, self.latent_spatial_dim_top, self.latent_spatial_dim_top),
                device=device,
                dtype=torch.long,
            )
            # Random bottom indices [B, H_b, W_b]
            code_b = torch.randint(
                0,
                self.num_embeddings,
                (num_samples, self.latent_spatial_dim_bottom, self.latent_spatial_dim_bottom),
                device=device,
                dtype=torch.long,
            )
            generated_samples = self.decode_code(code_t, code_b)
            
        return generated_samples

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        # Ensure all model parameters are on the same device
        self.to(self.device)

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
            self.vq_top._summary_mode = True
            self.vq_bottom._summary_mode = True
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
            self.vq_top._summary_mode = False
            self.vq_bottom._summary_mode = False
            self.train(was_training)

