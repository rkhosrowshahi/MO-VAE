"""Generate VAE and GG-VAE configs from VQ-VAE template structure."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "cifar100"
KLD_WEIGHT = 0.00512

# VQ-VAE structure: aggregator folders and their aggregator values
AGGREGATORS = [
    ("sum", "sum"),
    ("mgda", "mgda"),
    ("mgda_ln", "mgda_ln"),
    ("mgda_gn", "mgda_gn"),
    ("mgda_lgn", "mgda_lgn"),
    ("upgrad", "upgrad"),
    ("amtl_median", "aligned_mtl_median"),
    ("amtl_min", "aligned_mtl"),
]

# Objective configs: (folder, recons_objective, recons_activation, normalize_inputs)
OBJECTIVES = [
    ("mse", "mse", "none", True),
    ("bce", "bce", "sigmoid", False),
]

SEEDS = [
    (1, 42),
    (2, 123123),
    (3, 12341234),
]

# VAE config template - uses latent_dim, kld_loss
def vae_config(arch, aggregator_val, agg_folder, recons_obj, recons_act, norm, seed_val, seed_idx):
    name_part = f"cifar100-{arch}-128d-{recons_obj}-{agg_folder}-seed{seed_val}"
    return f"""dataset: cifar100
data_dir: ../data
normalize_inputs: {str(norm).lower()}
arch: {arch}
latent_dim: 128
hidden_dims:
- 32
- 64
- 128
- 256
- 512
loss_weights:
  reconstruction_loss: 1.0
  kld_loss: {KLD_WEIGHT}
recons_objective: {recons_obj}
recons_activation: {recons_act}
hv_ref:
  reconstruction_loss: 1.1
  kld_loss: 1.1
epochs: 200
batch_size: 256
optimizer: adam
lr: 1e-4
scheduler: cosine
scheduler_lr_min: 1e-6
wd: 0.0
aggregator: {aggregator_val}
seed: {seed_val}
save_path: logs/
save_freq: 20
eval_freq: 20
num_vis_samples: 9
use_wandb: true
wandb_project: mo-vae
wandb_entity: rasa_research
wandb_name: {name_part}
wandb_group: cifar100-{arch}-128d-{recons_obj}-{agg_folder}
"""


# GG-VAE config template - adds gradient_guided_loss, edge_matching_loss
def gg_vae_config(arch, aggregator_val, agg_folder, recons_obj, recons_act, norm, seed_val, seed_idx):
    name_part = f"cifar100-{arch}-128d-{recons_obj}-{agg_folder}-seed{seed_val}"
    return f"""dataset: cifar100
data_dir: ../data
normalize_inputs: {str(norm).lower()}
arch: {arch}
latent_dim: 128
hidden_dims:
- 32
- 64
- 128
- 256
- 512
loss_weights:
  reconstruction_loss: 1.0
  kld_loss: {KLD_WEIGHT}
  gradient_guided_loss: 1.0
  edge_matching_loss: 1.0
recons_objective: {recons_obj}
recons_activation: {recons_act}
hv_ref:
  reconstruction_loss: 1.1
  kld_loss: 1.1
  gradient_guided_loss: 1.1
  edge_matching_loss: 1.1
epochs: 200
batch_size: 256
optimizer: adam
lr: 1e-4
scheduler: cosine
scheduler_lr_min: 1e-6
wd: 0.0
aggregator: {aggregator_val}
seed: {seed_val}
save_path: logs/
save_freq: 20
eval_freq: 20
num_vis_samples: 9
use_wandb: true
wandb_project: mo-vae
wandb_entity: rasa_research
wandb_name: {name_part}
wandb_group: cifar100-{arch}-128d-{recons_obj}-{agg_folder}
"""


def main():
    # Architectures to create: vae, gg_vae, gg_vae_v2, ..., gg_vae_v6
    vae_archs = ["vae"]
    gg_vae_archs = ["gg_vae"] + [f"gg_vae_v{i}" for i in range(2, 7)]

    for arch in vae_archs:
        mse_paths = []
        bce_paths = []
        for agg_folder, agg_val in AGGREGATORS:
            for obj_folder, recons_obj, recons_act, norm in OBJECTIVES:
                for seed_idx, seed_val in SEEDS:
                    cfg_path = CONFIGS_DIR / arch / agg_folder / obj_folder / f"config_{seed_idx}.yaml"
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    content = vae_config(arch, agg_val, agg_folder, recons_obj, recons_act, norm, seed_val, seed_idx)
                    cfg_path.write_text(content, encoding="utf-8")
                    rel = str(cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if obj_folder == "mse":
                        mse_paths.append(rel)
                    else:
                        bce_paths.append(rel)

        (CONFIGS_DIR / arch / "mse_configs.txt").write_text("\n".join(mse_paths) + "\n", encoding="utf-8")
        (CONFIGS_DIR / arch / "bce_configs.txt").write_text("\n".join(bce_paths) + "\n", encoding="utf-8")

        # amtl_median, amtl_min subdir config lists
        for sub in ["amtl_median", "amtl_min"]:
            agg_folder, agg_val = [(a, v) for a, v in AGGREGATORS if a == sub][0]
            for obj_folder, recons_obj, recons_act, norm in OBJECTIVES:
                paths = []
                for seed_idx, seed_val in SEEDS:
                    cfg_path = CONFIGS_DIR / arch / sub / obj_folder / f"config_{seed_idx}.yaml"
                    paths.append(str(cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
                name = f"{obj_folder}_configs.txt"
                (CONFIGS_DIR / arch / sub / name).write_text("\n".join(paths) + "\n", encoding="utf-8")

    for arch in gg_vae_archs:
        mse_paths = []
        bce_paths = []
        for agg_folder, agg_val in AGGREGATORS:
            for obj_folder, recons_obj, recons_act, norm in OBJECTIVES:
                for seed_idx, seed_val in SEEDS:
                    cfg_path = CONFIGS_DIR / arch / agg_folder / obj_folder / f"config_{seed_idx}.yaml"
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    content = gg_vae_config(arch, agg_val, agg_folder, recons_obj, recons_act, norm, seed_val, seed_idx)
                    cfg_path.write_text(content, encoding="utf-8")
                    rel = str(cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if obj_folder == "mse":
                        mse_paths.append(rel)
                    else:
                        bce_paths.append(rel)

        (CONFIGS_DIR / arch / "mse_configs.txt").write_text("\n".join(mse_paths) + "\n", encoding="utf-8")
        (CONFIGS_DIR / arch / "bce_configs.txt").write_text("\n".join(bce_paths) + "\n", encoding="utf-8")

        for sub in ["amtl_median", "amtl_min"]:
            agg_folder, agg_val = [(a, v) for a, v in AGGREGATORS if a == sub][0]
            for obj_folder, recons_obj, recons_act, norm in OBJECTIVES:
                paths = []
                for seed_idx, seed_val in SEEDS:
                    cfg_path = CONFIGS_DIR / arch / sub / obj_folder / f"config_{seed_idx}.yaml"
                    paths.append(str(cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
                name = f"{obj_folder}_configs.txt"
                (CONFIGS_DIR / arch / sub / name).write_text("\n".join(paths) + "\n", encoding="utf-8")

    print("Generated configs for: vae, gg_vae, gg_vae_v2, gg_vae_v3, gg_vae_v4, gg_vae_v5, gg_vae_v6")


if __name__ == "__main__":
    main()
