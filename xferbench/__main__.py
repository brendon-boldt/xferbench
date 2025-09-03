import argparse
import sys
from pathlib import Path
import os

# Ensure that multiple GPUs are not visible since HuggingFace will
# automatically use mutliple GPUs and cause problems.  This must run before
# torch is loaded (directly or indirectly).
_cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_devs.split(",")[0]

import torch

from . import run, analysis

# Improvement: Using something typed
parser = argparse.ArgumentParser()
parser.add_argument("command", type=str)
parser.add_argument("--source", "-s", type=str)
parser.add_argument("--target", "-t", type=str)
parser.add_argument("--unit-test", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--danish-only", action="store_true")
parser.add_argument("--base-path", type=str)
parser.add_argument("--save-prefix", "--save", type=str)
parser.add_argument("--extra-name", type=str)
parser.add_argument("--result-path", type=str)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    help="Specify the config name as found in /model/config.py",
)
parser.add_argument(
    "--cpu-ok",
    action="store_true",
    help="Disable assertion that XferBench trains models on GPUs.",
)


def assert_cuda(rc: run.RunConfig) -> None:
    if not rc.cpu_ok and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; refusing to run models on CPU. Pass --cpu-ok to run on CPU anyway."
        )


_args = parser.parse_args()
rc = run.RunConfig(**vars(_args))

command_modules = [run, analysis]

for cm in command_modules:
    if hasattr(cm, rc.command):
        if cm == run:
            assert_cuda(rc)
        getattr(cm, rc.command)(rc)
        sys.exit(0)

if Path(rc.command).exists():
    assert_cuda(rc)
    print(f'No command specified; running full benchmark on "{rc.command}"...')
    run.benchmark(rc)
else:
    raise ValueError(f'"{rc.command}" does not correspond to a command or a file.')
