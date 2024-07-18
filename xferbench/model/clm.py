from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from tqdm import tqdm  # type: ignore
import torch
import transformers  # type: ignore
import tokenizers  # type: ignore

from . import config, common


ModelType = TypeVar("ModelType", bound=config.Model)


@common.check_complete
def init_model(*, model_cfg: config.Clm, data_path: Path) -> None:
    raw_dataset = common.get_raw_dataset(data_path)

    # If we're using the Wikipedia dataset
    if "title" in raw_dataset.features:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 10_000)))

    raw_dataset = raw_dataset.shuffle(seed=0)

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
        if "text" in raw_dataset.features:
            data_iterator = iter(raw_dataset["text"])
        else:
            data_iterator = iter([])

        tokenizer = tokenizers.ByteLevelBPETokenizer()
        reference_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        n_special_tokens = len(
            {t for t in reference_tokenizer.special_tokens_map.values()}
        )
        tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=model_cfg.vocab_size - n_special_tokens,
            min_frequency=2,
        )

        tokenizer.enable_truncation(model_cfg.context_length)
        tokenizer.save(str(model_cfg.tokenizer_path))


@common.check_complete
def train_base_model(*, model_cfg: config.Clm, data_path: Path | None) -> None:
    hf_config = transformers.GPT2Config(
        vocab_size=model_cfg.vocab_size,
        n_head=model_cfg.n_head,
        n_layer=model_cfg.n_layer,
        n_positions=model_cfg.context_length,
    )
    model = transformers.GPT2LMHeadModel(config=hf_config)

    if data_path is None:
        model.save_pretrained(str(model_cfg.save_path / "checkpoint-0"))
        return

    tokenizer = load_tokenizer(model_cfg)
    raw_dataset = common.get_raw_dataset(data_path)
    raw_dataset.shuffle(seed=0)

    # If we're dealing with the Wikipedia dataset
    if "title" in raw_dataset.features:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 15_000)))

    dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.train_dataset_size,
        save_path=model_cfg.save_path,
        cache=True,
        fail_on_too_small=False,
    )

    data_collator = get_data_collator(tokenizer)

    training_args = make_training_args(model_cfg, "train")

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    common.train_try_checkpoint(trainer)


@common.check_complete
def tune_model(
    *,
    model_cfg: config.Clm,
    data_path: Path,
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)
    tokenizer = load_tokenizer(model_cfg)

    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path)

    model.lm_head.reset_parameters()

    raw_dataset = common.get_raw_dataset(data_path)
    raw_datset = raw_dataset.train_test_split(seed=0, test_size=0.1)["train"]

    dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.tune_dataset_size,
    )
    data_collator = get_data_collator(tokenizer)

    training_args = make_training_args(model_cfg, "tune")

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    common.train_try_checkpoint(trainer)


@common.check_complete
def compute_model_score(*, data_path: Path, model_cfg: config.Clm) -> None:
    result_file = model_cfg.save_path / "result.txt"

    checkpoint_path = common.get_checkpoint(model_cfg.load_path)

    tokenizer = load_tokenizer(model_cfg)
    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.train(False)
    raw_dataset = common.get_raw_dataset(data_path)
    raw_dataset = raw_dataset.train_test_split(seed=0, test_size=0.1)["test"]
    raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 10_000)))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    batch_encodings = tokenizer(raw_dataset["text"])
    target_tokens = 1_000_000

    def gen():
        total = 0
        for be in batch_encodings[:]:
            yield torch.tensor(be.ids)
            yield torch.tensor([tokenizer.eos_token_id])
            total += len(be.ids) + 1
            if total > target_tokens:
                break

    dataset = torch.cat(list(gen())).unsqueeze(0)

    max_length = model.config.n_positions
    stride = 1 << 7
    seq_len = dataset.size(-1)

    model.to(device)

    nlls: list[Any] = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = dataset[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_ned_loc = end_loc
        if end_loc == seq_len:
            break

    cross_entropy = torch.stack(nlls).mean()

    print("cross-entropy", cross_entropy.item())

    with result_file.open("w") as fo:
        fo.write(str(cross_entropy.item()))


def load_tokenizer(mcfg: config.Clm) -> transformers.GPT2TokenizerFast:
    tokenizer = transformers.GPT2TokenizerFast(tokenizer_file=str(mcfg.tokenizer_path))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_data_collator(tokenizer: Any) -> Any:
    return transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


def make_training_args(cfg: config.Model, stage: Literal["train", "tune"]) -> Any:
    kwargs: dict[str, Any]
    match stage:
        case "train":
            kwargs = {
                "learning_rate": cfg.train_learning_rate,
                "num_train_epochs": cfg.n_train_epochs,
            }
        case "tune":
            kwargs = {
                "learning_rate": cfg.tune_learning_rate,
                "num_train_epochs": cfg.n_tune_epochs,
            }

    return transformers.TrainingArguments(
        output_dir=str(cfg.save_path),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        prediction_loss_only=True,
        logging_steps=cfg.logging_steps,
        **kwargs,
    )
