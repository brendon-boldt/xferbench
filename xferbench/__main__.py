import argparse
import sys
from pathlib import Path
import os

# Ensure that multiple GPUs are not visible since HuggingFace will
# automatically use mutliple GPUs and cause problems.
_cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_devs.split(",")[0]

from . import run, analysis

# Improvement: Using something typed
parser = argparse.ArgumentParser()
parser.add_argument("command", type=str)
parser.add_argument("--source", "-s", type=str)
parser.add_argument("--target", "-t", type=str)
parser.add_argument("--unit-test", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--save-prefix", "--save", type=str)
parser.add_argument("--config", "-c", type=str)

_args = parser.parse_args()
rc = run.RunConfig(**vars(_args))

command_modules = [run, analysis]

for cm in command_modules:
    if hasattr(cm, rc.command):
        getattr(cm, rc.command)(rc)
        sys.exit(0)

if Path(rc.command).exists():
    print(f'No command specified; running full benchmark on "{rc.command}"...')
    run.benchmark(rc)
else:
    raise ValueError(f'"{rc.command}" does not correspond to a command or a file.')
