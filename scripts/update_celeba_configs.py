#!/usr/bin/env python3
"""Temporary script: set epochs=400, save_freq=50, eval_freq=50 in all celeba-hq configs."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELEBA_CONFIGS = ROOT / "configs" / "celeba-hq"

REPLACEMENTS = [
    (re.compile(r"^(\s*epochs:\s*)\d+\s*$", re.MULTILINE), r"\g<1>400"),
    (re.compile(r"^(\s*save_freq:\s*)\d+\s*$", re.MULTILINE), r"\g<1>50"),
    (re.compile(r"^(\s*eval_freq:\s*)\d+\s*$", re.MULTILINE), r"\g<1>50"),
]


def main():
    updated = 0
    for path in CELEBA_CONFIGS.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for pattern, repl in REPLACEMENTS:
            new_text = pattern.sub(repl, new_text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            updated += 1
            print(path.relative_to(ROOT))
    print(f"\nUpdated {updated} config(s).")


if __name__ == "__main__":
    main()
