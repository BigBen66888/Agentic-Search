#!/usr/bin/env python3
"""Apply the minimal Volta-safe veRL patches in a reproducible way."""
import argparse
import os
import shutil
import site
from pathlib import Path

def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"pattern not found: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True

def patch_verl(source):
    source = Path(source)
    candidates = list(source.rglob("fsdp_workers.py"))
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        text = text.replace("attn_implementation='flash_attention_2'", "attn_implementation=os.environ.get('VERL_ATTN_IMPL', 'flash_attention_2')")
        if "import os" not in text.splitlines()[:30]:
            text = "import os\n" + text
        path.write_text(text, encoding="utf-8")
    rollout = next(iter(source.rglob("vllm_rollout_spmd.py")), None)
    if rollout:
        text = rollout.read_text(encoding="utf-8")
        if "_VERL_PREFIX_CACHING" not in text:
            text = "import os\n_VERL_PREFIX_CACHING = os.environ.get('VERL_VLLM_PREFIX_CACHING', '1').lower() in ('1','true','yes')\n" + text
            text = text.replace("enable_prefix_caching=True", "enable_prefix_caching=_VERL_PREFIX_CACHING")
            rollout.write_text(text, encoding="utf-8")
    print(f"[Search-R1] veRL V100 patch applied: {source}")

def install_shim():
    target = Path(site.getsitepackages()[0]) / "flash_attn"
    source = Path(__file__).resolve().parents[1] / "patches" / "flash_attn"
    target.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.py"):
        shutil.copy2(file, target / file.name)
    print(f"[Search-R1] flash_attn compatibility shim installed: {target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-src")
    parser.add_argument("--install-flash-shim", action="store_true")
    args = parser.parse_args()
    if args.verl_src:
        patch_verl(args.verl_src)
    if args.install_flash_shim:
        install_shim()
