import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

from utils.objectives import mse_per_image_sum, mse_per_pixel_mean, mse_total_batch_sum_scaled, bce_per_image_sum, bce_per_pixel_mean, laplacian_per_image_sum, laplacian_per_pixel_mean, kl_divergence


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, input, size=128):
        return input.view(-1, size, 4, 4)

# Define the GGVAE model
class GGVAE(nn.Module):
    def __init__(self, 
                 latent_dim=2, 
                 input_size=32, 
                 in_channels=3, 
                 hidden_dims=None, 
                 layer_norm="batch", 
                 output_activation="tanh", 
                 recons_dist="gaussian", recons_reduction="mean", lambda_weights=None, device=None, **kwargs):
        super(GGVAE, self).__init__()
        
        self.device = device
        
        recon_obj = None
        kld_obj = kl_divergence
        
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
        self.kld_obj = kld_obj

        sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0.,  2.],
                            [-1,  0.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)
        # Ensuring sobel filters can apply to RGB images - register as buffers so they move with model
        self.register_buffer('sobel_x', sobel_x.expand(3, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.expand(3, 1, 3, 3))

        self.objectives = {"reconstruction_loss": recon_obj, "kld_loss": kld_obj, "gradient_guided_loss": self.edge_weighted_pixel_loss, "edge_matching_loss": self.edge_matching_loss}

        self.features = ["mu", "log_var"]

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            lambda_weights = {"reconstruction_loss": 1.0, "kld_loss": 0.00025, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict: [reconstruction_weight, gradient_guided_weight, kld_weight]
            if len(lambda_weights) != 4:
                raise ValueError(f"GGVAE requires 4 lambda_weights (reconstruction, gradient_guided, kld), got {len(lambda_weights)}")
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
        
        self.lambda_weights = lambda_weights
        
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # Calculate spatial dimensions after len(hidden_dims) stride-2 convolutions
        # With stride=2, padding=1, kernel=3: each layer halves the spatial dimension
        # After n convolutions: input_size -> input_size//(2^n)
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        num_layers = len(hidden_dims)
        spatial_dim = input_size // (2 ** num_layers)
        encoder_output_size = hidden_dims[-1] * spatial_dim * spatial_dim

        if layer_norm == "batch":
            layer_norm = nn.BatchNorm2d
        elif layer_norm == "layer":
            layer_norm = nn.LayerNorm
        elif layer_norm == "none":
            layer_norm = nn.Identity
        else:
            raise ValueError(f"Layer norm {layer_norm} not supported")

        if output_activation == "tanh":
            output_activation = nn.Tanh
        elif output_activation == "sigmoid":
            output_activation = nn.Sigmoid
        elif output_activation == "none":
            output_activation = nn.Identity
        else:
            raise ValueError(f"Output activation {output_activation} not supported")
        
        # Build Encoder
        modules = []
        encoder_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    layer_norm(h_dim),
                    nn.LeakyReLU())
            )
            encoder_in_channels = h_dim
        
        modules.append(nn.Flatten())  # Flatten before linear layers
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.mu = nn.Linear(encoder_output_size, latent_dim)  # mean of the latent space
        self.log_var = nn.Linear(encoder_output_size, latent_dim)  # log variance of the latent space
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)
        
        modules = []
        hidden_dims_reversed = hidden_dims.copy()
        hidden_dims_reversed.reverse()
        
        # Add Unflatten layer
        modules.append(nn.Unflatten(1, (hidden_dims[-1], spatial_dim, spatial_dim)))
        
        # Build decoder layers (reverse of encoder)
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i],
                                       hidden_dims_reversed[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    layer_norm(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU())
            )
        
        # Final layer: last transpose conv + final conv + activation
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims_reversed[-1],
                               hidden_dims_reversed[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            layer_norm(hidden_dims_reversed[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims_reversed[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            output_activation()
        )
        
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        # Encode the input image
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterize the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decode the latent code
        out = self.decoder_input(z)
        out = self.decoder(out)
        recons = self.final_layer(out)
        return recons

    def forward(self, x):
        # Forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return {"recons": recons, "mu": mu, "log_var": log_var, "z": z}

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

    def edge_matching_loss(self, inputs, recons):
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
        recons = args["recons"]
        mu = args["mu"]
        log_var = args["log_var"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        gradient_guided_loss = self.objectives["gradient_guided_loss"](inputs, recons)
        kld_loss = self.objectives["kld_loss"](mu, log_var)
        
        # Apply lambda_weights using dictionary keys matching self.objectives
        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_gradient_guided_loss = self.lambda_weights["gradient_guided_loss"] * gradient_guided_loss
        weighted_kld_loss = self.lambda_weights["kld_loss"] * kld_loss

        return {"reconstruction_loss": weighted_recon_loss, "gradient_guided_loss": weighted_gradient_guided_loss, "kld_loss": weighted_kld_loss}

    def sample(self, num_samples=1, device=None):
        """
        Sample from the latent space and generate images.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated_samples = self.decode(z)
        return generated_samples

    def print_model_summary(self):
        device = self.device
        if device.type != 'cuda' or device.type != 'cuda:0':
            device = 'cuda'
        return summary(self, (self.in_channels, self.input_size, self.input_size), device=device)