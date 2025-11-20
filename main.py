import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchjd.autojac import backward, mtl_backward
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
    Sum
)
import wandb
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import scienceplots
from torchvision.utils import make_grid

from utils.utils import AverageMeter, get_dataset, set_seed
from models import get_network

plt.style.use(["science", "ieee", "no-latex"])

# Global step tracker for wandb logging in hooks
_current_step = 0


def print_weights(_, __, weights: torch.Tensor) -> None:
    """Prints the extracted weights."""
    global _current_step
    # print(f"Weights: {weights}")
    if wandb.run is not None:
        log_dict = {f"train/task_{i}_weight": weight.item() for i, weight in enumerate(weights)}
        wandb.log(log_dict, step=_current_step)


def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], weights: torch.Tensor) -> None:
    """Prints the cosine similarity between the weighted aggregation and the average gradient."""
    global _current_step
    matrix = inputs[0]
    # Compute mean gradient (simple average across tasks)
    gd_output = matrix.mean(dim=0)
    # Compute weighted aggregated gradient
    # matrix shape: [num_tasks, num_params], weights shape: [num_tasks]
    weighted_grad = (matrix.T @ weights)  # Shape: [num_params]
    similarity = torch.nn.functional.cosine_similarity(weighted_grad, gd_output, dim=0)
    similarity_value = similarity.item()
    # print(f"Cosine similarity: {similarity_value:.4f}")
    if wandb.run is not None:
        wandb.log({"train/gradient_similarity": similarity_value}, step=_current_step)


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

        print(f"Total loss: {total_loss.item():.6e}")

        # Verify decoder gradients match between mtl_backward and standard backward
        # This ensures that KLD (which doesn't use decoder) doesn't affect decoder gradients via MTL
        # if step == 0 and aggregator is not None and aggregator != "sum":
        #     features_check = None
        #     if "mu" in outputs and "log_var" in outputs:
        #         features_check = [outputs["mu"], outputs["log_var"]]
        #     elif "encoding" in outputs:
        #         features_check = [outputs["encoding"]]

        #     if features_check is not None:
        #         print("Verifying decoder gradients integrity...")
                
        #         # 1. Compute gradients with standard sum backward
        #         optimizer.zero_grad()
        #         total_loss.backward(retain_graph=True)
                
        #         decoder_grads = {}
        #         decoder_net = net.module.decoder if hasattr(net, "module") else net.decoder
        #         for name, param in decoder_net.named_parameters():
        #             if param.grad is not None:
        #                 decoder_grads[name] = param.grad.clone()

        #         # 2. Compute gradients with mtl_backward
        #         optimizer.zero_grad()
        #         mtl_backward(
        #             losses=list(loss_dict.values()),
        #             features=features_check,
        #             aggregator=aggregator,
        #             retain_graph=True 
        #         )

        #         # 3. Compare gradients
        #         max_diff = 0.0
        #         for name, param in decoder_net.named_parameters():
        #             if name in decoder_grads:
        #                 grad_sum = decoder_grads[name]
        #                 grad_mtl = param.grad
        #                 if grad_mtl is not None and grad_sum is not None:
        #                     diff = (grad_mtl - grad_sum).abs().max().item()
        #                     max_diff = max(max_diff, diff)
                
        #         print(f"Max decoder gradient difference: {max_diff:.6e}")
        #         if max_diff < 1e-5:
        #             print("Gradient Check Passed: Decoder gradients are consistent.")
        #         else:
        #             print("Gradient Check Failed: Decoder gradients differ!")
                
        #         optimizer.zero_grad()

        # Update global step for hooks before backward
        global _current_step
        _current_step = step + 1
        
        if aggregator is None or aggregator == "sum":
            total_loss.backward()
        else:
            features = None
            if net.features is not None:
                features = [outputs[feature] for feature in net.features]
            
            if features is not None:
                mtl_backward(
                    losses=list(loss_dict.values()),
                    features=features,
                    aggregator=aggregator,
                )
            else:
                backward(loss_dict.values(), aggregator=aggregator)

        # Clip gradients to prevent numerical instabilities
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_grad_norm)

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


def evaluate(net, data_loader, device, args):
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

    train_dataset, test_dataset, input_size = get_dataset(args.dataset, normalize=args.normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    args.dataset_size = len(train_dataset)

    net = get_network(input_size, num_channels=3, args=args).to(device)
    args.total_params = net.total_trainable_params()
    if hasattr(net, "print_model_summary"):
        print(net.print_model_summary())

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
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.scheduler_lr_min)
        elif args.scheduler == "multi_step":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)
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
            aggregator = CAGrad(c=1.0)
        elif agg_name == "nashmtl":
            aggregator = NashMTL(n_tasks=len(net.objectives))
        elif agg_name == "dualproj":
            aggregator = DualProj()
        elif agg_name == "jd_sum":
            aggregator = Sum()
        elif agg_name == "sum":
            aggregator = "sum"
        else:
            raise ValueError(f"Aggregator {args.aggregator} not supported")
    else:
        args.aggregator = "sum"

    if aggregator is not None and aggregator != "sum":
        aggregator.weighting.register_forward_hook(print_weights)
        aggregator.weighting.register_forward_hook(print_gd_similarity)

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

    # eval_loss_meters = evaluate(net, train_loader, device=device, args=args)
    # print(
    #     "Initial random loss: "
    #     + ", ".join(f"{key}: {meter.avg:.6e}" for key, meter in eval_loss_meters.items())
    # )

    objective_keys = net.objectives.keys()
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
        eval_loss_meters = evaluate(net, test_loader, device=device, args=args)

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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--aggregator", "--agg", type=str, default=None)
    parser.add_argument("--arch", type=str, default="vae")
    parser.add_argument("--layer_norm", type=str, default="batch")
    parser.add_argument("--output_activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--objs", type=str, nargs="+", default=["mse_mean", "kld"])
    parser.add_argument("--kld_weight", type=float, default=0.00025)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--scheduler_lr_min", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_milestones", type=int, nargs="+", default=None)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--anneal_steps", type=int, default=200)
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
