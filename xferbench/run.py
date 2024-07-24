import json
from typing import Literal, cast, overload, Generic, TypeVar, Any
from pathlib import Path
import re
import shutil

import pydantic
import torch

from .model import config, mt, clm, mlm


class PydanticModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())


class RunConfig(PydanticModel):
    """Interface for executing an action.

    This is inteded to be useable from the command line via argparse,
    programmatically via Python, or from JSON config files.
    """

    command: str = "default"
    source: str | None = None
    target: str | None = None
    overwrite: bool = False
    unit_test: bool = False
    save_prefix: str | None = None
    config: str | None = None
    cpu_ok: bool = False


ModelClass = TypeVar("ModelClass", bound=config.Model)


class RunEnvironment(PydanticModel, Generic[ModelClass]):
    """Computed values based on the provided `RunConfig`."""

    base_save_path: Path
    base_data_path: Path
    model_class: type[ModelClass]


Task = Literal["clm", "mt", "mlm"]


@overload
def get_env(rc: RunConfig, task: Literal["clm"]) -> RunEnvironment[config.Clm]:
    ...


@overload
def get_env(rc: RunConfig, task: Literal["mlm"]) -> RunEnvironment[config.Mlm]:
    ...


@overload
def get_env(rc: RunConfig, task: Literal["mt"]) -> RunEnvironment[config.Mt]:
    ...


def get_env(rc: RunConfig, task: Task) -> RunEnvironment:
    base_path = Path("unit-test/" if rc.unit_test else ".")

    if rc.save_prefix is None:
        match task:
            case "clm":
                suffix = "save-clm"
            case "mt":
                suffix = "save-mt"
            case "mlm":
                suffix = "save-mlm"
            case _:
                raise ValueError()
        base_save_path = base_path / suffix
    else:
        base_save_path = Path(rc.save_prefix)

    if rc.config is None:
        model_classes: tuple[type[config.Model], type[config.Model]]
        match task:
            case "clm":
                model_classes = (config.Clm, config.ClmTest)
            case "mt":
                model_classes = (config.Mt, config.MtTest)
            case "mlm":
                # Maybe make a test config later
                model_classes = (config.Mlm, config.Mlm)
        model_class = model_classes[int(rc.unit_test)]
    else:
        model_class = getattr(config, rc.config)

    return RunEnvironment(
        base_data_path=base_path / "data",
        base_save_path=base_save_path,
        model_class=model_class,
    )


def clm_init_model(rc: RunConfig) -> None:
    env = get_env(rc, "clm")

    assert rc.source is None and rc.target is not None

    save_path = env.base_save_path / f"{rc.target}-tokenizer"

    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.save()
    data_path = env.base_data_path / "eval" / rc.target

    clm.init_model(model_cfg=model_cfg, data_path=data_path)


def clm_train_base_model(rc: RunConfig) -> None:
    env = get_env(rc, "clm")
    assert rc.source is not None and rc.target is None

    save_path = env.base_save_path / rc.source / "train"
    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.save()

    if rc.source in ["no_pt"]:
        data_path = None
    else:
        data_path = env.base_data_path / "baselines" / rc.source
        clm.init_model(model_cfg=model_cfg, data_path=data_path)

    clm.train_base_model(
        model_cfg=model_cfg,
        data_path=data_path,
    )


def clm_tune_model(rc: RunConfig) -> None:
    env = get_env(rc, "clm")
    assert rc.source is not None and rc.target is not None

    load_path = env.base_save_path / rc.source / "train"
    save_path = env.base_save_path / rc.source / f"{rc.target}-{rc.config}-tune"
    data_path = env.base_data_path / "eval" / rc.target
    tokenizer_path = env.base_save_path / f"{rc.target}-tokenizer" / "tokenizer.json"

    tune_model_cfg = env.model_class(
        load_path=load_path,
        save_path=save_path,
        tokenizer_path=tokenizer_path,
    )
    tune_model_cfg.save()

    score_model_cfg = tune_model_cfg.model_copy()
    score_model_cfg.load_path = tune_model_cfg.save_path
    score_model_cfg.init_dirs()

    clm.tune_model(
        model_cfg=tune_model_cfg,
        data_path=data_path,
    )

    clm.compute_model_score(
        model_cfg=score_model_cfg,
        data_path=data_path,
    )


def mt_train_base_model(rc: RunConfig) -> None:
    env = get_env(rc, "mt")

    lang = rc.source
    assert lang is not None

    save_path = env.base_save_path / lang / "train"
    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.save()

    if lang in ["no_pt"]:
        data_path = None
    else:
        data_path = env.base_data_path / "baselines" / lang

    mt.train_base_model(model_cfg=model_cfg, data_path=data_path)


def mt_init_model(rc: RunConfig) -> None:
    env = get_env(rc, "mt")
    assert rc.source is None and rc.target is not None

    _target_pair = tuple(rc.target.split("-"))
    assert len(_target_pair) == 2
    target_pair = cast(tuple[str, str], _target_pair)

    ds_name = env.model_class.model_fields["task_dataset"].default
    save_path = env.base_save_path / f"{rc.target}-{ds_name}-tokenizer"
    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.save()
    mt.init_model(
        model_cfg=model_cfg,
        target_pair=target_pair,
    )


def mt_tune_model(rc: RunConfig) -> None:
    env = get_env(rc, "mt")

    assert rc.source is not None and rc.target is not None
    _target_pair = tuple(rc.target.split("-"))
    assert len(_target_pair) == 2
    target_pair = cast(tuple[str, str], _target_pair)

    ds_name = env.model_class.model_fields["task_dataset"].default
    freeze = env.model_class.model_fields["finetune_frozen"].default
    tune_load_path = env.base_save_path / rc.source / "train"
    tune_save_path = (
        env.base_save_path / rc.source / f"{rc.target}-{env.model_class.__name__}-tune"
    )
    if freeze:
        tune_save_path = Path(str(tune_save_path) + "-freeze")

    tokenizer_path = (
        env.base_save_path / f"{rc.target}-{ds_name}-tokenizer" / "tokenizer.json"
    )

    tune_model_cfg = env.model_class(
        load_path=tune_load_path,
        save_path=tune_save_path,
        tokenizer_path=tokenizer_path,
    )
    tune_model_cfg.save()

    score_model_cfg = tune_model_cfg.model_copy()
    score_model_cfg.load_path = tune_model_cfg.save_path
    score_model_cfg.init_dirs()

    mt.tune_model(
        model_cfg=tune_model_cfg,
        target_pair=target_pair,
    )

    mt.compute_score(
        save_path=tune_save_path,
        model_cfg=score_model_cfg,
        target_pair=target_pair,
    )


def mlm_train_base_model(rc: RunConfig) -> None:
    raise NotImplementedError()
    env = get_env(rc, "mlm")

    lang = rc.source
    assert lang is not None

    save_path = env.base_save_path / lang / "train"
    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.init_dirs()
    if rc.source in ["no_pt", "no_pttn"]:
        data_path = None
    else:
        data_path = env.base_data_path / "baselines" / lang
    mlm.train_base_model(model_cfg=model_cfg, data_path=data_path)


def mlm_init_model(rc: RunConfig) -> None:
    raise NotImplementedError()
    env = get_env(rc, "mlm")
    assert rc.source is None and rc.target is not None

    save_path = env.base_save_path / f"{rc.target}-tokenizer"
    data_path = env.base_data_path / f"eval/{rc.target}"
    model_cfg = env.model_class(
        save_path=save_path,
        tokenizer_path=save_path / "tokenizer.json",
    )
    model_cfg.init_dirs()
    mlm.init_model(
        model_cfg=model_cfg,
        data_path=data_path,
    )


def mlm_tune_model(rc: RunConfig) -> None:
    raise NotImplementedError()
    env = get_env(rc, "mlm")

    assert rc.source is not None and rc.target is not None

    mlm_load_path = env.base_save_path / rc.source / "train"
    if rc.source in ["no_tn", "no_pttn"]:
        data_path = None
    else:
        data_path = env.base_data_path / f"eval/{rc.target}"
    tune_prefix = f"{rc.config}-tune"
    mlm_save_path = env.base_save_path / rc.source / f"{tune_prefix}-mlm"

    tokenizer_path = env.base_save_path / f"{rc.target}-tokenizer" / "tokenizer.json"

    mlm_model_cfg = env.model_class(
        load_path=mlm_load_path,
        save_path=mlm_save_path,
        tokenizer_path=tokenizer_path,
    )
    mlm_model_cfg.init_dirs()

    task_model_cfg = mlm_model_cfg.model_copy()
    task_model_cfg.load_path = mlm_model_cfg.save_path
    task_model_cfg.save_path = env.base_save_path / rc.source / f"{tune_prefix}-task"
    task_model_cfg.save()

    score_model_cfg = task_model_cfg.model_copy()
    score_model_cfg.load_path = task_model_cfg.save_path

    mlm.tune_model_mlm(
        model_cfg=mlm_model_cfg,
        data_path=data_path,
    )

    mlm.tune_model_task(
        model_cfg=task_model_cfg,
    )

    mlm.compute_score(
        model_cfg=score_model_cfg,
    )


def check_eval_langs(base_data_path: Path) -> None:
    paths = [base_data_path / "eval" / l for l in config.target_languages]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Cannot find evaluation language at data {p}. Ensure that you have run xferbench/scripts/wikipedia.py to download the evaluation languages."
            )


def benchmark(rc: RunConfig) -> None:
    if rc.command == "benchmark":
        assert rc.source is not None
        source_data_path = Path(rc.source)
    else:
        source_data_path = Path(rc.command)

    env = get_env(rc, "clm")
    check_eval_langs(env.base_data_path)

    name = f"{source_data_path.parents[0].name}_{source_data_path.stem}"

    base_source_save_path = env.base_save_path / f"xferbench-{name}"

    train_save_path = base_source_save_path / "train"
    base_model_cfg = env.model_class(
        save_path=train_save_path, tokenizer_path=train_save_path / "tokenizer.json"
    )
    base_model_cfg.save()

    clm.init_model(
        model_cfg=base_model_cfg,
        data_path=source_data_path,
    )
    clm.train_base_model(
        model_cfg=base_model_cfg,
        data_path=source_data_path,
    )
    for el in config.target_languages:
        tune_tokenizer_path = env.base_save_path / f"{el}-tokenizer" / "tokenizer.json"
        tune_model_cfg = env.model_class(
            tokenizer_path=tune_tokenizer_path,
            save_path=base_source_save_path / f"{el}-tune",
            load_path=base_model_cfg.save_path,
        )
        tune_model_cfg.save()
        clm.init_model(
            model_cfg=tune_model_cfg,
            data_path=env.base_data_path / "eval" / el,
        )
        tune_data_path = data_path = env.base_data_path / "eval" / el
        clm.tune_model(
            model_cfg=tune_model_cfg,
            data_path=tune_data_path,
        )
        score_save_path = base_source_save_path / f"{el}-result"
        score_save_path.mkdir(parents=True, exist_ok=True)

        score_model_cfg = tune_model_cfg.model_copy()
        score_model_cfg.load_path = tune_model_cfg.save_path
        score_model_cfg.init_dirs()

        clm.compute_model_score(
            model_cfg=score_model_cfg,
            data_path=tune_data_path,
        )

        for p in tune_model_cfg.save_path.glob("checkpoint-*"):
            shutil.rmtree(p)

    summary: dict[str, Any] = {"by_target": {}}
    for path in base_source_save_path.glob("**/result.txt"):
        with path.open() as fo:
            matches = re.search(r"/([^/]+)-tune/", str(path))
            assert matches is not None
            lang = matches[1]
            summary["by_target"][lang] = float(fo.read())
    scores = list(summary["by_target"].values())
    summary["score"] = sum(scores) / len(scores)
    print()
    print(f"XferBench score: {summary['score']:.3f}")
    with (base_source_save_path / "final-score.txt").open("w") as fo:
        fo.write(str(summary["score"]))
    results_path = base_source_save_path / "results.json"
    with results_path.open("w") as fo:
        json.dump(summary, fo, indent=2)
    print(f'Results summary in "{results_path}".')
