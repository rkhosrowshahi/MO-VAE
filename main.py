import os
from argparse import ArgumentParser
from collections import defaultdict
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
from tqdm import tqdm

from models import VAE, VAE3
from optim import AlignedMTLBalancer, LinearScalarization


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse_recons_loss_sum(data, logits):
    reconstructed_data = logits[0]
    loss = F.mse_loss(reconstructed_data, data, reduction="sum")
    return loss


def mse_recons_loss(data, logits):
    reconstructed_data = logits[0]
    loss = F.mse_loss(reconstructed_data, data, reduction="mean")
    return loss


def bce_recons_loss(data, logits):
    reconstructed_data = logits[0]
    loss = F.binary_cross_entropy_with_logits(reconstructed_data, data, reduction="sum")
    return loss


# Beta-KL divergence loss
def bkl_loss(data, logits):
    mu, log_var = logits[1], logits[2]

    beta = args.beta
    # loss = beta * -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())
    kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
    loss = beta * kl
    return loss


def eval_step(
    model, train_dataset, batch_size, optimizer, balancer, device, train_loader=None
):
    if train_loader is None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criteria = {"reconstruction": mse_recons_loss_sum, "kl": bkl_loss}
    # Train the VAE model
    model.train()
    avg_total_loss, avg_task_losses, avg_task_weights, avg_comp_metrics = (
        0,
        defaultdict(float),
        defaultdict(float),
        defaultdict(float),
    )
    for batch_idx, (data, _) in enumerate(train_loader):

        balancer.step_with_model(data=data.to(device), model=model, criteria=criteria)
        optimizer.step()
        losses = balancer.losses
        loss_weights = balancer.loss_weights
        info = balancer.info
        avg_total_loss += sum(
            losses[task_id] * loss_weights[task_id] for task_id in losses
        )
        for task_id in losses:
            avg_task_losses[task_id] += losses[task_id]
            avg_task_weights[task_id] += loss_weights[task_id]

        for metric_id in info:
            avg_comp_metrics[metric_id] += info[metric_id]

    avg_total_loss = avg_total_loss / len(train_loader)
    for task_id in avg_task_losses:
        avg_task_losses[task_id] /= len(train_loader)
        avg_task_weights[task_id] /= len(train_loader)

    for metric_id in avg_comp_metrics:
        avg_comp_metrics[metric_id] /= len(train_loader)

    return avg_total_loss, avg_task_losses, avg_task_weights, avg_comp_metrics


def train_step(
    model, optimizer, balancer, device, train_metrics=None, train_loader=None
):
    criteria = {"reconstruction": mse_recons_loss_sum, "kl": bkl_loss}
    # Train the VAE model
    model.train()
    loss_total, task_losses, task_weights = 0, defaultdict(float), defaultdict(float)
    for batch_idx, (data, _) in enumerate(train_loader):

        balancer.step_with_model(data=data.to(device), model=model, criteria=criteria)
        optimizer.step()
        losses = balancer.losses
        loss_weights = balancer.loss_weights
        # if hasattr(balancer, 'info') and balancer.info is not None:
        #     fmtl_metrics.write(utils.strfy(balancer.info) + "\n")
        #     fmtl_metrics.flush()
        loss_total += sum(losses[task_id] * loss_weights[task_id] for task_id in losses)
        for task_id in losses:
            task_losses[task_id] += losses[task_id]
            task_weights[task_id] += loss_weights[task_id]

        train_metrics.append(
            {
                "train_loss": loss_total,
                "task_losses": losses,
                "task_weights": loss_weights,
            }
        )
    avg_total_loss = loss_total / len(train_loader)
    for task_id in task_losses:
        task_losses[task_id] /= len(train_loader)
        task_weights[task_id] /= len(train_loader)
    return avg_total_loss, task_losses, task_weights


def plot_after_training(model, data_loader, train_metrics, save_path, device, beta):
    model.eval()
    test_input, test_label = next(iter(data_loader))
    test_input = test_input.to(device)
    test_label = test_label.to(device)

    vutils.save_image(
        test_input.data,
        os.path.join(save_path, "figures/original_images.png"),
        normalize=True,
        nrow=12,
    )

    reconstructed_data = model(test_input)[0]
    vutils.save_image(
        reconstructed_data.data,
        os.path.join(save_path, "figures/generated_images.png"),
        normalize=True,
        nrow=12,
    )

    try:
        samples = model.sample(144, device, labels=test_label)
        vutils.save_image(
            samples.cpu().data,
            os.path.join(save_path, "figures/sampled_images.png"),
            normalize=True,
            nrow=12,
        )
    except Warning:
        pass

    # data = data.to(device)
    # reconstructed_data = model(data)[0]

    # orig_img = torchvision.utils.make_grid(data, nrow=n)
    #         orig_img = np.transpose(orig_img.cpu().numpy(), (1, 2, 0))

    #         recon_img = torchvision.utils.make_grid(reconstructed_data, nrow=n)
    #         recon_img = np.transpose(recon_img.cpu().numpy(), (1, 2, 0))

    # plt.imshow(orig_img)
    # plt.axis("off")
    # plt.title("Original Images")
    # plt.savefig(os.path.join(save_path, "figures/original_images.png"))
    # plt.savefig(os.path.join(save_path, "figures/original_images.pdf"))
    # plt.close()

    # plt.imshow(recon_img)
    # plt.axis("off")
    # plt.title("Generated Images")
    # plt.savefig(os.path.join(save_path, "figures/generated_images.png"))
    # plt.savefig(os.path.join(save_path, "figures/generated_images.pdf"))
    # plt.close()

    total_loss_values = np.array([entry["train_loss"] for entry in train_metrics])
    iterations = np.arange(1, len(total_loss_values) + 1, 100, dtype=int)
    plt.plot(iterations, total_loss_values[iterations - 1], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Total loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "figures/total_loss_plot.png"))
    plt.savefig(os.path.join(save_path, "figures/total_loss_plot.pdf"))
    plt.close()
    # plt.show()

    recons_loss_values = np.array(
        [entry["task_losses"]["reconstruction"] for entry in train_metrics]
    )
    plt.plot(iterations, recons_loss_values[iterations - 1], marker="s")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Reconstruction loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "figures/reconstruction_loss_plot.png"))
    plt.savefig(os.path.join(save_path, "figures/reconstruction_loss_plot.pdf"))
    plt.close()
    # plt.show()

    kl_loss_values = np.array([entry["task_losses"]["kl"] for entry in train_metrics])
    plt.plot(iterations, kl_loss_values[iterations - 1], marker="^")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Beta KL-divergence" + r"$(\beta={beta})$" + "loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "figures/kl_loss_plot.png"))
    plt.savefig(os.path.join(save_path, "figures/kl_loss_plot.pdf"))
    plt.close()

    # np.savez(
    #     os.path.join(save_path, "data/train_metrics.npz"),
    #     total_loss=total_loss_values,
    #     recons_loss=recons_loss_values,
    #     kl_loss=kl_loss_values,
    # )
    rec_loss_w = [entry["task_weights"]["reconstruction"] for entry in train_metrics]
    kl_loss_w = [entry["task_weights"]["kl"] for entry in train_metrics]

    df = pd.DataFrame(
        {
            "steps": np.arange(1, len(total_loss_values) + 1, dtype=int),
            "total_loss": total_loss_values,
            "rec_loss": recons_loss_values,
            "rec_weight": rec_loss_w,
            "kl_loss": kl_loss_values,
            "kl_weight": kl_loss_w,
        }
    )
    df.to_csv(os.path.join(save_path, "data/train_metrics.csv"), index=False)


def main(args):
    # Define the device (GPU or CPU)
    if torch.cuda.device_count() == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(device)

    in_channels = 0
    in_height = 0
    train_dataset, test_dataset = None, None
    model = None
    if args.dataset.lower() == "cifar10":
        # Load the CIFAR10 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        in_channels = 3
        in_height = 32
        # Initialize the VAE model
        model = VAE3(
            latent_dim=args.latent_dim,
            in_size=in_height,
            in_channels=in_channels,
            hidden_dims=[32, 64, 128, 256],
        ).to(device)
    elif args.dataset.lower() == "celeba":
        # Load the CelebA dataset
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.CelebA(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.CelebA(
            root="./data", split="test", download=True, transform=transform
        )
        in_channels = 3
        in_height = 128
        # Initialize the VAE model
        model = VAE3(
            latent_dim=args.latent_dim,
            in_size=in_height,
            in_channels=in_channels,
            hidden_dims=[32, 64, 128, 256, 512, 1024],
        ).to(device)
    elif args.dataset.lower() == "fashion":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                transforms.Normalize(0.5, 0.5),
            ]
        )
        # Load the Fashion dataset
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        in_channels = 1
        in_height = 28
        # Initialize the VAE model
        model = VAE(
            latent_dim=args.latent_dim, in_height=in_height, in_channels=in_channels
        ).to(device)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(
        f"Model total size: {sum(p.numel() for p in model.parameters())}, Encoder size: {sum(p.numel() for p in model.encoder.parameters())}, Latent repr size: {sum(p.numel() for p in model.mu.parameters())+sum(p.numel() for p in model.log_var.parameters())}, Decoder size: {sum(p.numel() for p in model.decoder.parameters())}"
    )
    # Create data loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=144, shuffle=False)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    balancer = None
    if args.optimizer == "multi":
        balancer = AlignedMTLBalancer(
            scale_mode=args.scaler, compute_stats=args.compute_stats
        )
    elif args.optimizer == "single":
        balancer = LinearScalarization(compute_stats=args.compute_stats)

    res_path = os.path.join(
        args.output_path,
        args.dataset.lower(),
        args.optimizer.lower(),
        args.scaler.lower(),
        str(args.batch_size) + "batchsize",
        str(args.latent_dim) + "latent",
        str(args.run),
    )
    print(res_path)
    os.makedirs(res_path + "/figures", exist_ok=True)
    os.makedirs(res_path + "/data", exist_ok=True)

    if args.compute_stats:
        print("Computing statistics...")

        train_metrics = []
        for epoch in tqdm(range(args.epochs)):
            avg_train_loss, avg_task_losses, avg_task_weights, avg_comp_metrics = (
                eval_step(
                    model,
                    train_dataset,
                    batch_size,
                    optimizer,
                    balancer,
                    device,
                    train_loader=train_loader,
                )
            )
            train_metrics.append(
                {
                    "train_loss": avg_train_loss,
                    "task_losses": avg_task_losses,
                    "stats": avg_comp_metrics,
                }
            )
        save_path = f"./outputs/{args.dataset}/{args.optimizer}_{args.scaler}-scaler_beta/{args.epochs}epochs_{args.batch_size}batchsize_{args.seed}seed/"
        # os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + "figures", exist_ok=True)
        os.makedirs(save_path + "data", exist_ok=True)

        total_loss_values = [entry["train_loss"] for entry in train_metrics]
        recons_loss_values = [
            entry["task_losses"]["reconstruction"] for entry in train_metrics
        ]
        kl_loss_values = [entry["task_losses"]["kl"] for entry in train_metrics]
        statistics = [entry["stats"] for entry in train_metrics]
        # np.savez(
        #     os.path.join(save_path, "data/train_metrics_with_stats.npz"),
        #     total_loss=total_loss_values,
        #     recons_loss=recons_loss_values,
        #     kl_loss=kl_loss_values,
        #     stats=statistics,
        # )
        df = pd.DataFrame(
            {
                "steps": range(1, args.epochs + 1),
                "total_loss": total_loss_values,
                "rec_loss": recons_loss_values,
                "kl_loss": kl_loss_values,
            }
        )
        df.to_csv(
            os.path.join(save_path, "data/train_metrics_with_stats.csv"), index=False
        )
    else:

        train_metrics = []
        for epoch in range(args.epochs):
            avg_train_loss, avg_task_losses, avg_task_weights = train_step(
                model,
                optimizer,
                balancer,
                device,
                train_metrics,
                train_loader=train_loader,
            )
            # Print the loss at each epoch
            print(f"Epoch: {epoch}, ", f"avg_train_loss: {avg_train_loss}, ", end=" ")
            for task_id in avg_task_losses:
                print(
                    "loss_{}: {:.6e}, weight_{}:{:.6f}".format(
                        task_id,
                        avg_task_losses[task_id],
                        task_id,
                        avg_task_weights[task_id],
                    ),
                    end=", ",
                )
            print()

            # print(f'curr_LR: {scheduler.get_last_lr()}')
            # Update the scheduler
            # scheduler.step()

        # res_path = f"./outputs/{args.dataset}/{args.optimizer}_{args.scaler}-scaler/{args.epochs}epochs_{args.batch_size}batchsize_{args.seed}seed/"
        # os.makedirs(save_path, exist_ok=True)
        # Save the checkpoint of model
        torch.save(model.state_dict(), os.path.join(res_path, "data/model_weights.pth"))
        # Evaluate model to generate images
        plot_after_training(
            model, test_loader, train_metrics, res_path, device, args.beta
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="multi")
    parser.add_argument(
        "--scaler", type=str, default="min", choices=["linear", "min", "median", "rmse"]
    )
    parser.add_argument("--lr", type=float, default="0.001")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_stats", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--output-path", type=str, default="./logs/", help="path to store the results"
    )
    parser.add_argument("--run", type=int, default=1, help="[1..10]")
    # parser.set_defaults(compute_stats=False)

    args = parser.parse_args()
    set_seed(args.seed + args.run)
    main(args)
