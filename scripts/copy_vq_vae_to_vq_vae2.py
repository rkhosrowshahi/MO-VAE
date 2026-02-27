#!/usr/bin/env python3
"""Temporary script: copy celeba-hq vq_vae configs to vq_vae2 and set arch/paths to vq_vae2."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "celeba-hq" / "vq_vae"
DST = ROOT / "configs" / "celeba-hq" / "vq_vae2"


def main():
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    for path in DST.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            # Replace vq_vae with vq_vae2 so arch, paths, wandb_name, wandb_group, comments get updated
            new_text = text.replace("vq_vae", "vq_vae2")
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                print(path.relative_to(ROOT))

    print(f"\nDone: copied {SRC.relative_to(ROOT)} -> {DST.relative_to(ROOT)} and set vq_vae -> vq_vae2.")


if __name__ == "__main__":
    main()
