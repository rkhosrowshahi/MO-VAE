import torch
import torch.nn as nn
from torch.nn import functional as F

from .vae import VAE

# Small epsilon value for numerical stability in gradient computations
EPS = 1e-8


# Define the GGVAE model
class GGVAE(VAE):
    """
    Gradient-Guided Variational Autoencoder (GGVAE) model
    
    Inherits from VAE and adds gradient-guided and edge matching losses.
    """
    def __init__(self, 
                 latent_dim=2, 
                 input_size=32, 
                 in_channels=3, 
                 hidden_dims=None, 
                 layer_norm="batch", 
                 output_activation="tanh", 
                 recons_dist="gaussian", 
                 recons_reduction="mean", 
                 lambda_weights=None, 
                 device=None, 
                 **kwargs):
        # Initialize parent VAE class
        super(GGVAE, self).__init__(
            latent_dim=latent_dim,
            input_size=input_size,
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            layer_norm=layer_norm,
            output_activation=output_activation,
            recons_dist=recons_dist,
            recons_reduction=recons_reduction,
            lambda_weights=None,  # We'll set this after adding additional objectives
            device=device,
            **kwargs
        )

        # Initialize Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0.,  2.],
                            [-1,  0.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)
        # Ensuring sobel filters can apply to RGB images - register as buffers so they move with model
        self.register_buffer('sobel_x', sobel_x.expand(3, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.expand(3, 1, 3, 3))

        # Update objectives dictionary to include gradient-guided and edge matching losses
        self.objectives = {"reconstruction_loss": self.recon_obj, "kld_loss": self.kld_obj, 
                          "gradient_guided_loss": self.edge_weighted_pixel_loss, 
                          "edge_matching_loss": self.edge_matching_loss}

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            lambda_weights = {"reconstruction_loss": 1.0, "kld_loss": 0.00025, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict: [reconstruction_weight, kld_weight, gradient_guided_weight, edge_matching_weight]
            if len(lambda_weights) != 4:
                raise ValueError(f"GGVAE requires 4 lambda_weights (reconstruction, kld, gradient_guided, edge_matching), got {len(lambda_weights)}")
            lambda_weights = {
                "reconstruction_loss": lambda_weights[0],
                "kld_loss": lambda_weights[1],
                "gradient_guided_loss": lambda_weights[2],
                "edge_matching_loss": lambda_weights[3]
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
        
        # Override lambda_weights from parent
        self.lambda_weights = lambda_weights

    # def gradient_guided_loss(self,inputs, recons):
    #     # Sobel applied to R,G,B
    #     x_grad = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
    #     y_grad = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))

    #     #Gradient magnitude
    #     grad_mag = torch.sqrt(x_grad**2 + y_grad**2)# (batch_size,C,H,W)

    #     # Combining across channels - take max across channels
    #     grad_max = torch.max(grad_mag, dim=1)[0]

    #     # Normalizing
    #     grad_max = (grad_max - grad_max.view(grad_max.size(0), -1).min(1, keepdim=True)[0].unsqueeze(-1)) / \
    #                         (grad_max.view(grad_max.size(0), -1).max(1, keepdim=True)[0].unsqueeze(-1) -
    #                         grad_max.view(grad_max.size(0), -1).min(1, keepdim=True)[0].unsqueeze(-1))

    #     # non-reduced reconstruction loss (B, C, H, W)
    #     pixel_loss = F.mse_loss(recons, inputs, reduction='none')

    #     #Gradient-guided Encoder Loss
    #     loss_grad = (grad_max.unsqueeze(1) * pixel_loss).mean() #(batch_size, C, H, W) then mean across batch

    #     return loss_grad

    def edge_weighted_pixel_loss(self, inputs, recons):
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge-weighted pixel loss: BCE (pixels are in [0,1])
        grad_target = torch.sqrt(input_x**2 + input_y**2 + EPS) #(batch_size,C,H,W)
        weights = grad_target.max(dim=1)[0]  # simplified
        weights = weights / (weights.max() + EPS)  # normalize to [0,1]
        
        pixel_loss = F.binary_cross_entropy(recons, inputs, reduction='none')  # BCE here
        weighted_pixel_loss = (weights.unsqueeze(1) * pixel_loss).mean()
        
        return weighted_pixel_loss

    def edge_matching_loss(self, inputs, recons):
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge matching: L1 loss (gradients are signed, unbounded)
        # Computes gradient magnitudes and compares them with L1
        # Purpose: Forces the model to reproduce edge structures
        grad_pred = torch.sqrt(recon_x**2 + recon_y**2 + EPS)
        grad_target = torch.sqrt(input_x**2 + input_y**2 + EPS)

        edge_match_loss = F.l1_loss(grad_pred, grad_target)
        
        return edge_match_loss

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu = args["mu"]
        log_var = args["log_var"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        gradient_guided_loss = self.objectives["gradient_guided_loss"](inputs, recons)
        edge_matching_loss = self.objectives["edge_matching_loss"](inputs, recons)
        kld_loss = self.objectives["kld_loss"](mu, log_var)
        
        # Apply lambda_weights using dictionary keys matching self.objectives
        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_gradient_guided_loss = self.lambda_weights["gradient_guided_loss"] * gradient_guided_loss
        weighted_edge_matching_loss = self.lambda_weights["edge_matching_loss"] * edge_matching_loss
        weighted_kld_loss = self.lambda_weights["kld_loss"] * kld_loss

        return {
            "reconstruction_loss": weighted_recon_loss, 
            "gradient_guided_loss": weighted_gradient_guided_loss,
            "edge_matching_loss": weighted_edge_matching_loss,
            "kld_loss": weighted_kld_loss
        }