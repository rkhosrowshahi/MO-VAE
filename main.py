import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchjd import backward, mtl_backward
from torchjd.aggregation import (
    UPGrad,
    PCGrad,
    Mean,
    AlignedMTL,
    NashMTL,
    IMTLG,
    MGDA,
    CAGrad,
    DualProj,
)
import wandb
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import scienceplots
from torchvision.utils import make_grid

from utils.utils import AverageMeter, get_dataset, set_seed
from models import get_network

plt.style.use(["science", "ieee", "no-latex"])


def train_epoch(net, train_loader, optimizer, aggregator, step, device, args):
    net.train()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()

    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()

        outputs = net(images)
        loss_dict = net.loss_function(images, args=outputs)
        total_loss = sum(loss_dict.values())

        if aggregator is None or aggregator == "none":
            total_loss.backward()
        else:
            features = None
            if "mu" in outputs and "log_var" in outputs:
                features = [outputs["mu"], outputs["log_var"]]
            if features is not None:
                mtl_backward(loss_dict.values(), features=features, aggregator=aggregator)
            else:
                backward(loss_dict.values(), aggregator=aggregator)

        optimizer.step()

        loss_meters["total_loss"].update(total_loss.item())
        for key, value in loss_dict.items():
            loss_meters[key].update(value.item())

        step += 1
        if args.use_wandb:
            wandb.log(
                {
                    **{f"train/{key}": meter.avg for key, meter in loss_meters.items()},
                    **{f"train/{key}_curr": meter.val for key, meter in loss_meters.items()},
                },
                step=step,
            )

    return loss_meters, step


def eval_epoch(net, data_loader, device, args):
    net.eval()
    loss_meters = {key: AverageMeter() for key in net.objectives.keys()}
    loss_meters["total_loss"] = AverageMeter()

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = net(images)
            loss_dict = net.loss_function(images, args=outputs)
            total_loss = sum(loss_dict.values())

            loss_meters["total_loss"].update(total_loss.item())
            for key, value in loss_dict.items():
                loss_meters[key].update(value.item())

    return loss_meters


def generate_random_samples(net, num_samples, device, save_path=None, log_to_wandb=False, epoch=None, step=None):
    samples = net.sample(num_samples=num_samples, device=device)
    grid = make_grid(samples, nrow=int(np.sqrt(num_samples)), normalize=True)

    if save_path is not None:
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))
        grid_np = np.clip(grid_np, 0, 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title("Generated Random Samples")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    if log_to_wandb and epoch is not None:
        wandb.log({"generated_samples": wandb.Image(grid, caption=f"Epoch {epoch}")}, step=step)

    return samples


def generate_reconstructed_samples(
    net,
    split_name,
    loader,
    num_samples,
    device,
    save_path=None,
    log_to_wandb=False,
    epoch=None,
    step=None,
):
    net.eval()
    originals, reconstructions = [], []

    with torch.no_grad():
        collected = 0
        for images, _ in loader:
            if collected >= num_samples:
                break

            images = images.to(device)
            take = min(images.size(0), num_samples - collected)
            batch = images[:take]

            outputs = net(batch)
            recon = outputs.get("recons")
            if recon is None:
                raise KeyError("Model output dictionary must contain 'recons'.")

            originals.append(batch.cpu())
            reconstructions.append(recon.cpu())
            collected += take

    originals = torch.cat(originals, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0)

    comparison = []
    for idx in range(num_samples):
        comparison.append(originals[idx])
        comparison.append(reconstructions[idx])
    comparison_tensor = torch.stack(comparison)
    grid = make_grid(comparison_tensor, nrow=int(np.sqrt(num_samples)) * 2, normalize=True)

    if save_path is not None:
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))
        grid_np = np.clip(grid_np, 0, 1)

        plt.figure(figsize=(15, 15))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title(f"Reconstructed {split_name} Samples (Original | Reconstructed)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    if log_to_wandb and epoch is not None:
        wandb.log(
            {f"reconstructed_{split_name}_samples": wandb.Image(grid, caption=f"Epoch {epoch}")},
            step=step,
        )

    return originals, reconstructions


def build_hv_indicator(objective_keys, args):
    if len(objective_keys) < 2:
        return None

    ref_point = []
    for key in objective_keys:
        if key == "reconstruction_loss":
            ref_point.append(args.hv_ref_recon if args.hv_ref_recon is not None else 1e6)
        elif key == "kl_loss":
            ref_point.append(args.hv_ref_kl if args.hv_ref_kl is not None else 1000.0)
        else:
            ref_point.append(1e6)

    return HV(ref_point=np.array(ref_point))


def main(args):
    device = torch.device(args.device)

    train_dataset, test_dataset, input_size, _, _ = get_dataset(args.dataset, normalize=args.normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    net = get_network(input_size, num_channels=3, args=args).to(device)
    args.total_params = net.total_trainable_params()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = None
    if args.scheduler is not None:
        if args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
        elif args.scheduler == "multi_step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        else:
            raise ValueError(f"Scheduler {args.scheduler} not supported")

    aggregator = None
    if args.aggregator is not None:
        agg_name = args.aggregator.lower()
        if agg_name == "upgrad":
            aggregator = UPGrad()
        elif agg_name == "pcgrad":
            aggregator = PCGrad()
        elif agg_name == "mean":
            aggregator = Mean()
        elif agg_name == "aligned_mtl":
            aggregator = AlignedMTL()
        elif agg_name == "imtlg":
            aggregator = IMTLG()
        elif agg_name == "mgda":
            aggregator = MGDA()
        elif agg_name == "cagrad":
            aggregator = CAGrad()
        elif agg_name == "nashmtl":
            aggregator = NashMTL()
        elif agg_name == "dualproj":
            aggregator = DualProj()
        elif agg_name == "none":
            aggregator = "none"
        else:
            raise ValueError(f"Aggregator {args.aggregator} not supported")
    else:
        args.aggregator = "none"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(args.save_path, args.dataset, args.arch, args.optimizer, args.aggregator, timestamp)
    os.makedirs(os.path.join(save_root, "figures", "generated"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "figures", "reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "checkpoints"), exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name if args.wandb_name else None,
            config=vars(args),
            dir=save_root,
            tags=args.wandb_tags if args.wandb_tags else None,
        )

    eval_loss_meters = eval_epoch(net, train_loader, device=device, args=args)
    print(
        "Initial random loss: "
        + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in eval_loss_meters.items())
    )

    objective_keys = [key for key in eval_loss_meters.keys() if key != "total_loss"]
    hv_indicator = build_hv_indicator(objective_keys, args)

    step = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss_meters, step = train_epoch(
            net,
            train_loader=train_loader,
            optimizer=optimizer,
            aggregator=aggregator,
            step=step,
            device=device,
            args=args,
        )
        eval_loss_meters = eval_epoch(net, test_loader, device=device, args=args)

        if (epoch % args.save_freq == 0) or epoch in {1, args.epochs}:
            gen_path = os.path.join(save_root, "figures", "generated", f"epoch_{epoch:03d}_random_samples.pdf")
            generate_random_samples(
                net,
                num_samples=args.num_samples,
                device=device,
                save_path=gen_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

            test_path = os.path.join(save_root, "figures", "reconstructed", f"epoch_{epoch:03d}_test_samples.pdf")
            generate_reconstructed_samples(
                net,
                "test",
                test_loader,
                args.num_samples,
                device,
                save_path=test_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

            train_path = os.path.join(save_root, "figures", "reconstructed", f"epoch_{epoch:03d}_train_samples.pdf")
            generate_reconstructed_samples(
                net,
                "train",
                train_loader,
                args.num_samples,
                device,
                save_path=train_path,
                log_to_wandb=args.use_wandb,
                epoch=epoch,
                step=step,
            )

        if hv_indicator is not None:
            train_point = np.array([[train_loss_meters[key].avg for key in objective_keys]])
            eval_point = np.array([[eval_loss_meters[key].avg for key in objective_keys]])
            train_hv = hv_indicator(train_point)
            eval_hv = hv_indicator(eval_point)
        else:
            train_hv = float("nan")
            eval_hv = float("nan")

        log_dict = {
            "epoch": epoch,
            **{f"train/{key}": meter.avg for key, meter in train_loss_meters.items()},
            **{f"eval/{key}": meter.avg for key, meter in eval_loss_meters.items()},
            "train/hv": train_hv,
            "eval/hv": eval_hv,
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        if args.use_wandb:
            wandb.log(log_dict, step=step)

        if scheduler is not None:
            scheduler.step()

        tqdm.write(
            f" Epoch {epoch}/{args.epochs} - Train - "
            + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in train_loss_meters.items())
            + f", HV: {train_hv:.2e}"
        )
        tqdm.write(
            f" Epoch {epoch}/{args.epochs} - Eval - "
            + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in eval_loss_meters.items())
            + f", HV: {eval_hv:.2e}"
        )

    tqdm.write("Training completed!")
    final_ckpt = os.path.join(save_root, "checkpoints", "final_checkpoint.pth")
    torch.save(net.state_dict(), final_ckpt)

    if args.use_wandb:
        try:
            wandb.save(final_ckpt)
        except (OSError, PermissionError):
            try:
                artifact = wandb.Artifact("final_model", type="model")
                artifact.add_file(final_ckpt)
                wandb.log_artifact(artifact)
            except Exception:
                wandb.log({"final_checkpoint_path": final_ckpt})
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="logs/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aggregator", "--agg", type=str, default=None)
    parser.add_argument("--arch", type=str, default="vae")
    parser.add_argument("--layer_norm", type=str, default="batch")
    parser.add_argument("--output_activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--objs", type=str, nargs="+", default=["mse_sum", "kl_sum"])
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", type=int, nargs="+", default=None)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hv_ref_recon", type=float, default=None)
    parser.add_argument("--hv_ref_kl", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mo-vae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)

    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    main(args)
