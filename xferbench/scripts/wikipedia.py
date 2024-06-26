from pathlib import Path
import argparse

import datasets  # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument(
    "which",
    type=str,
    choice=["eval", "all"],
    help='Which data split to download; either "eval" for the languages necessary to run XferBench or "all" for the eval languages and the baseline natural languages.',
)
args = ap.parse_args()


ds_path = "wikimedia/wikipedia"
make_ds_name = lambda l: f"20231101.{l}"
match args.which:
    case "eval":
        baseline_langs = []
    case "all":
        baseline_langs = ["fr", "es", "ru", "zh", "ar", "hi", "ko"]
    case _:
        raise ValueError()

eval_langs = ["da", "eu", "fa", "fi", "he", "id", "ja", "kk", "ro", "ur"]


for lang in baseline_langs + eval_langs:
    print(lang)
    ds = datasets.load_dataset(
        ds_path, make_ds_name(lang), revision="97323c5edeffcf4bd6786b4ed0788c84abd24b03"
    )["train"]
    ds.shuffle(seed=0)
    if lang in baseline_langs:
        if lang != "hi":
            ds = ds.select(range(200_000))
    else:
        ds = ds.select(range(100_000))

    print(f"{lang}: {sum(len(x) for x in ds['text'])/1e6:.1f}M")
    if lang in baseline_langs:
        save_path = Path("data/baselines")
    else:
        save_path = Path("data/eval")
    save_path.mkdir(exist_ok=True, parents=True)
    ds.save_to_disk(str(save_path / lang))
