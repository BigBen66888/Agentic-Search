#!/usr/bin/env python3
import importlib.util
import os
import subprocess
import sys

def check(name):
    ok = importlib.util.find_spec(name) is not None
    print(f"[Search-R1] {name}: {'OK' if ok else 'MISSING'}", flush=True)
    return ok

print("[Search-R1] 环境检查开始", flush=True)
print("[Search-R1] Python:", sys.version.replace("\n", " "), flush=True)
for package in ["datasets", "pandas", "pyarrow", "transformers", "faiss", "ray", "verl"]:
    check(package)
if check("torch"):
    import torch
    print(f"[Search-R1] torch={torch.__version__}", flush=True)
    print(f"[Search-R1] cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[Search-R1] gpu_count={torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"[Search-R1] gpu[{i}]={torch.cuda.get_device_name(i)}", flush=True)
if importlib.util.find_spec("ray"):
    subprocess.run(["ray", "status"], check=False)
print("[Search-R1] 环境检查结束", flush=True)

