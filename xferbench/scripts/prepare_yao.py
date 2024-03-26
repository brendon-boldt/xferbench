from pathlib import Path
import json

import requests  # type: ignore
import torch


yao_ec_file = Path("./cc.pt")
yao_ec_dest_file = Path("./data/baselines/yao/data.jsonl")
url_yao_ec = "https://drive.google.com/u/3/uc?id=1rMAWBLEu0R3mqsEJvnU1IrS5d4scivBI&export=download&confirm=yes"

paren_real_file = Path("paren-zipf.pt")
paren_real_dest_file = Path("./data/baselines/pz_real/data.jsonl")
url_paren_real = "https://drive.google.com/u/3/uc?id=15MgZTPY-lOYbeXSmxK-ii6ESqTfZFu2N&export=download&confirm=yes"


def download(url: str, save_path: Path) -> None:
    if not save_path.exists():
        res = requests.get(url)
        with save_path.open("wb") as fo:
            fo.write(res.content)


if __name__ == "__main__":
    download(url_yao_ec, yao_ec_file)
    yao_ec = torch.load(yao_ec_file)
    yao_ec_dest_file.parents[0].mkdir(parents=True, exist_ok=True)
    with yao_ec_dest_file.open("w") as fo:
        for line in yao_ec:
            json.dump(line.tolist(), fo)
            fo.write("\n")

    VOCAB_SIZE = 30000 - 10
    download(url_paren_real, paren_real_file)
    paren_real = torch.load(paren_real_file)[0]
    paren_real = paren_real[paren_real < VOCAB_SIZE]
    paren_real_dest_file.parents[0].mkdir(parents=True, exist_ok=True)
    with paren_real_dest_file.open("w") as fo:
        json.dump(paren_real.tolist(), fo)
