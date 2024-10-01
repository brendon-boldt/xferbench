from glob import glob
from pathlib import Path
from typing import Any, Callable, TypeVar, cast
import tempfile
import json

import tokenizers  # type: ignore
import transformers  # type: ignore
import datasets  # type: ignore

from . import config

Func = TypeVar("Func", bound=Callable)


# Improvement: hash parameters to determine if we need to rerun things.
def check_complete(func: Func) -> Func:
    def wrapper(
        model_cfg: config.Model | None = None,
        save_path: Path | None = None,
        **kwargs,
    ) -> None:
        comp_fn = f"{func.__name__}.complete"
        comp_save_path = None
        if save_path:
            comp_save_path = save_path
        elif model_cfg:
            comp_save_path = model_cfg.save_path
        if comp_save_path is None:
            raise ValueError(
                "Function which uses this wrapper must either have save_path or a ModelConfig with a save path."
            )
        comp_path = comp_save_path / comp_fn

        save_kwargs: dict[str, Any] = {}
        if save_path:
            save_kwargs["save_path"] = save_path
        if model_cfg:
            save_kwargs["model_cfg"] = model_cfg

        if not comp_path.exists():
            func(**save_kwargs, **kwargs)
            comp_save_path.mkdir(parents=True, exist_ok=True)
            comp_path.touch()
        else:
            print(f"Found {comp_path}, skipping execution.")

    return cast(Func, wrapper)


def get_checkpoint(model_source_base: str | Path) -> str:
    checkpoints = glob(f"{model_source_base}/checkpoint-*")
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {model_source_base}.")
    key = lambda s: int(s.split("-")[-1])
    return sorted(checkpoints, key=key)[-1]


def get_encoder(tokenizer: Any, block_size: int) -> Callable:
    # Ensure that any parameters of get_encoder inside of encode (i.e.,
    # closures) will hash the same so as not to break cachine.
    def encode(examples: Any) -> dict:
        if "text" in examples:
            examples = tokenizer(
                examples["text"],
                return_special_tokens_mask=False,
                return_length=True,
                # max_length=block_size,
                truncation=False,
            )
        elif "source_text" in examples:
            examples = tokenizer(
                examples["source_text"],
                text_target=examples["target_text"],
                return_special_tokens_mask=True,
                return_length=True,
                max_length=block_size,
                truncation=True,
                padding=False,
            )
        task_is_lm = "labels" not in examples

        if task_is_lm:
            _examples = {}
            _examples["input_ids"] = examples["input_ids"]
            _examples["length"] = [len(x) for x in examples["input_ids"]]
            examples = _examples

        return examples

    return encode


def get_tokenized_dataset(
    model_cfg: config.Model,
    raw_dataset: datasets.arrow_dataset.Dataset,
    *,
    tokenizer: Any,
    n_tokens_target: int | None,
    save_path: Path | None = None,
    cache: bool = False,
    fail_on_too_small: bool = True,
    no_blocks: bool = False,
) -> Any:
    """

    Parameters
    ----------
    n_tokens_target: int
        the length of the returned dataset will be as close to
        `n_tokens_target` as possible while being at least that large
    no_blocks: bool, default False
        Do not group the dataset into blocks.  Useful if you want to analyze
        the length of lines, say.
    """

    if cache:
        if save_path is None:
            raise ValueError("save_path must be set if cache is True.")
        tokenized_path = str(save_path / "tokenized_dataset.hf")
        try:
            return datasets.load_from_disk(tokenized_path)
        except FileNotFoundError:
            pass

    block_size = model_cfg.context_length
    # This is just the number of examples per batch during the dataset.map()
    # operation.  It is not related to any hyperparameters in the model.
    map_batch_size = 1000

    task_is_lm = False

    encode = get_encoder(tokenizer, block_size)
    dataset = raw_dataset.map(
        encode, batched=True, remove_columns=list(raw_dataset.features.keys())
    )

    if n_tokens_target is not None:
        dataset_length = sum(dataset["length"])
        if dataset_length < n_tokens_target:
            if fail_on_too_small:
                raise RuntimeError(
                    f"Dataset was not large enough to satisfy target token requirement ({dataset_length} < {n_tokens_target})"
                )

            # During processing, at most block_size - 1 will be truncated per batch.
            max_truncated_tokens = (len(dataset) // map_batch_size + 1) * (
                block_size - 1
            )
            # Assume the worst case scenario where every batch truncates (block_size - 1) tokens.
            n_repeats = n_tokens_target // (dataset_length - max_truncated_tokens) + 1
            print(f"WARNING: dataset too small; repeating {n_repeats} times")
            ds_list = [dataset.shuffle(seed=i) for i in range(n_repeats)]
            dataset = datasets.concatenate_datasets(ds_list)

    def to_blocks(examples) -> dict:
        if "length" in examples:
            del examples["length"]
        flatten = lambda arr: [x for y in arr for x in y]
        concatenated_examples: dict = {k: flatten(examples[k]) for k in examples.keys()}

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = total_length - (total_length % block_size)
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        result["attention_mask"] = [[1] * block_size] * len(result["input_ids"])
        result["length"] = [block_size] * len(result["input_ids"])
        return result

    if "labels" not in dataset.features and not no_blocks:
        # We're doing language modeling, so we can concatenate to block size
        # to maximize effeciency.
        _dataset = dataset
        dataset = dataset.map(
            to_blocks,
            batched=True,
            remove_columns=list(dataset.features.keys()),
            batch_size=map_batch_size,
        )

    if n_tokens_target is not None:
        # Binary search to find the number of data to match the number of tokens.
        lo = 0
        hi = len(dataset)
        lengths = dataset["length"]
        while hi != lo:
            idx = (hi + lo) // 2
            if sum(lengths[:idx]) < n_tokens_target:
                lo = idx + 1
            else:
                hi = idx
        dataset = dataset.select(range(hi))

    if not (n_tokens_target is None or sum(dataset["length"]) >= n_tokens_target):
        raise ValueError(
            "Dataset was not large enough after repetition. The input dataset was likely not repeated enough times. This is an implementation error."
        )

    if cache:
        dataset.save_to_disk(tokenized_path)
    return dataset


def train_try_checkpoint(trainer: transformers.Trainer) -> None:
    run_wo_chkpt = False
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        run_wo_chkpt = True

    if run_wo_chkpt:
        trainer.train()


def get_raw_dataset(path: Path) -> datasets.arrow_dataset.Dataset:
    try:
        return datasets.load_from_disk(path)
    except FileNotFoundError as e:
        pass
        print(e)
    return _get_raw_dataset_from_lines(path)


def _get_raw_dataset_from_lines(path: Path) -> datasets.arrow_dataset.Dataset:
    tfo: None | tempfile._TemporaryFileWrapper = None

    if path.is_dir():
        path = next(path.glob("*.jsonl"))

    with path.open() as fo:
        data = json.loads(fo.readline())
        # Eventually, we should make that utterances cannot be length zero or
        # handle this properly
        dtype = type(data[0] if len(data) else 0)
        if not isinstance(data, dict):
            tfo = tempfile.NamedTemporaryFile("w")
            path = Path(tfo.name)
            fo.seek(0)
            field = "input_ids" if dtype == int else "text"
            while line := fo.readline():
                tfo.write(f'{{"{field}":{line.rstrip()}}}\n')
            tfo.flush()

    try:
        return datasets.load_dataset(
            "json",
            data_files=str(path),
        )["train"]
    except Exception as e:
        raise e
    finally:
        if tfo is not None:
            tfo.close()


TokenizerClass = TypeVar("TokenizerClass", bound=transformers.PreTrainedTokenizerFast)


def load_tokenizer(model_cfg: config.Model) -> Any:
    return model_cfg.tokenizer_class(tokenizer_file=str(model_cfg.tokenizer_path))


def make_tokenizer(
    *,
    model_cfg: config.Model,
    raw_dataset: datasets.arrow_dataset.Dataset,
) -> None:
    need_tokenizer = True
    try:
        load_tokenizer(model_cfg)
        print(f"Tokenizer found at {model_cfg.tokenizer_path}")
        need_tokenizer = False
    except Exception:
        # Hugging Face uses a generic exception, so we can't be more
        # specific with the `except`.
        pass

    if need_tokenizer:
        # dataset = get_raw_dataset(data_path)
        if "text" in raw_dataset.features:
            data_iterator = iter(raw_dataset["text"])
        else:
            data_iterator = iter([])

        tokenizer = tokenizers.ByteLevelBPETokenizer()

        match model_cfg.tokenizer_class:
            case transformers.BartTokenizerFast:
                ref_name = "facebook/bart-base"
            case transformers.AlbertTokenizerFast:
                ref_name = "albert-base-v2"

        reference_tokenizer = model_cfg.tokenizer_class.from_pretrained(ref_name)
        n_special_tokens = len(
            {t for t in reference_tokenizer.special_tokens_map.values()}
        )
        tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=model_cfg.vocab_size - n_special_tokens,
            min_frequency=2,
        )
        tokenizer.save(str(model_cfg.tokenizer_path))
