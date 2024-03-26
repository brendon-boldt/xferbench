from pathlib import Path
from typing import Any, Literal
import functools

import torch
import transformers  # type: ignore
import tokenizers  # type: ignore
import datasets  # type: ignore
import evaluate  # type: ignore
import numpy as np

from . import config, common
from ..external import dlm_collator


def make_tokenizer(
    *,
    model_cfg: config.Model,
    raw_dataset: datasets.arrow_dataset.Dataset,
) -> None:
    need_tokenizer = True
    try:
        load_tokenizer(model_cfg.tokenizer_path)
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
        reference_tokenizer = transformers.BartTokenizerFast.from_pretrained(
            "facebook/bart-base"
        )
        n_special_tokens = len(
            {t for t in reference_tokenizer.special_tokens_map.values()}
        )
        tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=model_cfg.vocab_size - n_special_tokens,
            min_frequency=2,
        )
        tokenizer.save(str(model_cfg.tokenizer_path))


def make_training_args(
    mcfg: config.Mt, phase: Literal["train", "tune"]
) -> transformers.Seq2SeqTrainingArguments:
    kwargs: dict[str, Any]
    match phase:
        case "train":
            kwargs = {
                "learning_rate": mcfg.train_learning_rate,
                "num_train_epochs": mcfg.n_train_epochs,
            }
        case "tune":
            kwargs = {
                "learning_rate": mcfg.tune_learning_rate,
                "evaluation_strategy": "steps",
                "eval_steps": 1 / 20,
                "num_train_epochs": mcfg.n_tune_epochs,
            }

    return transformers.Seq2SeqTrainingArguments(
        output_dir=mcfg.save_path,
        per_device_train_batch_size=mcfg.per_device_train_batch_size,
        per_device_eval_batch_size=mcfg.per_device_train_batch_size,
        weight_decay=0.01,
        save_total_limit=mcfg.save_total_limit,
        predict_with_generate=True,
        save_steps=mcfg.save_steps,
        logging_steps=mcfg.logging_steps,
        fp16=torch.cuda.is_available(),
        **kwargs,
    )


def get_tune_data(
    lang_pair: tuple[str, str], dataset_name: str
) -> datasets.DatasetDict:
    pair_str = "-".join(lang_pair)
    rev_pair_str = "-".join(reversed(lang_pair))
    try:
        raw_dataset = datasets.load_dataset(dataset_name, pair_str)
    except ValueError:
        raw_dataset = datasets.load_dataset(dataset_name, rev_pair_str)

    return raw_dataset["train"].train_test_split(test_size=0.2, seed=0)


@common.check_complete
def init_model(*, model_cfg: config.Mt, target_pair: tuple[str, str]) -> None:
    raw_dataset = get_tune_data(target_pair, model_cfg.task_dataset)["train"]

    # This is appropriate for the WMT dataset
    raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 100_000)))

    def to_text(examples: Any) -> Any:
        examples["text"] = [
            ex[target_pair[0]] + " " + ex[target_pair[1]]
            for ex in examples["translation"]
        ]
        return examples

    raw_dataset = raw_dataset.map(to_text, batched=True)

    make_tokenizer(
        model_cfg=model_cfg,
        raw_dataset=raw_dataset,
    )


@common.check_complete
def train_base_model(*, model_cfg: config.Mt, data_path: Path | None) -> None:
    hf_model_config = transformers.BartConfig(
        vocab_size=model_cfg.vocab_size,
        encoder_ffn_dim=model_cfg.encoder_ffn_dim,
        decoder_ffn_dim=model_cfg.decoder_ffn_dim,
        d_model=model_cfg.d_model,
        max_position_embeddings=model_cfg.max_position_embeddings,
        encoder_layers=model_cfg.encoder_layers,
        decoder_layers=model_cfg.decoder_layers,
        encoder_attention_heads=model_cfg.encoder_attention_heads,
        decoder_attention_heads=model_cfg.decoder_attention_heads,
    )
    model = transformers.BartForConditionalGeneration(config=hf_model_config)

    if data_path is None:
        model.save_pretrained(str(model_cfg.save_path / "checkpoint-0"))
        return

    raw_dataset = common.get_raw_dataset(data_path)
    raw_dataset = raw_dataset.shuffle(seed=0)

    make_tokenizer(
        model_cfg=model_cfg,
        raw_dataset=raw_dataset,
    )
    tokenizer = load_tokenizer(model_cfg.tokenizer_path)

    # If we're dealing with the Wikipedia dataset
    if "title" in raw_dataset.features:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 150_000)))

    dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        save_path=model_cfg.save_path,
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.train_dataset_size,
        fail_on_too_small=False,
    )
    data_collator = dlm_collator.DataCollatorForBartDenoisingLM(
        tokenizer=tokenizer,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    training_args = make_training_args(model_cfg, "train")

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    common.train_try_checkpoint(trainer)


@common.check_complete
def tune_model(
    *,
    model_cfg: config.Mt,
    target_pair: tuple[str, str],
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)

    tokenizer = load_tokenizer(model_cfg.tokenizer_path)
    raw_dataset = get_tune_data(target_pair, model_cfg.task_dataset)
    raw_train_ds = raw_dataset["train"].select(
        range(min(len(raw_dataset["train"]), 2_000_000))
    )
    raw_val_ds = raw_dataset["test"].select(range(min(len(raw_dataset["test"]), 1000)))

    def split_translation(example: dict) -> dict:
        return {
            "source_text": example["translation"][target_pair[0]],
            "target_text": example["translation"][target_pair[1]],
        }

    raw_train_ds = raw_train_ds.map(split_translation, remove_columns=["translation"])
    raw_val_ds = raw_val_ds.map(split_translation, remove_columns=["translation"])

    train_dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_train_ds,
        save_path=model_cfg.save_path,
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.tune_dataset_size,
        cache=True,
    )
    val_dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_val_ds,
        tokenizer=tokenizer,
        n_tokens_target=None,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=model_cfg.max_position_embeddings,
    )

    model = transformers.BartForConditionalGeneration.from_pretrained(checkpoint_path)
    if model_cfg.finetune_frozen:
        model.train(False)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
        model.lm_head.train(True)

    # All of the embedding matrices are tied.
    model.lm_head.reset_parameters()

    training_args = make_training_args(model_cfg, "tune")

    compute_metrics = functools.partial(_compute_metrics, tokenizer)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    common.train_try_checkpoint(trainer)


def _compute_metrics(tokenizer, eval_preds) -> dict[str, Any]:
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels) -> tuple[Any, Any]:
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    _result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    _result_chrf = chrf.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": _result["score"], "chrf": _result_chrf["score"]}
    return result


@common.check_complete
def compute_score(
    *,
    model_cfg: config.Mt,
    save_path: Path,
    target_pair: tuple[str, str],
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)
    model = transformers.BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.train(False)

    raw_dataset = get_tune_data(target_pair, model_cfg.task_dataset)["test"]
    raw_dataset = raw_dataset.select(range(min(len(raw_dataset), 5000)))

    def split_translation(example: dict) -> dict:
        return {
            "source_text": example["translation"][target_pair[0]],
            "target_text": example["translation"][target_pair[1]],
        }

    raw_dataset = raw_dataset.map(split_translation, remove_columns=["translation"])

    tokenizer = load_tokenizer(model_cfg.tokenizer_path)
    dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        save_path=save_path,
        tokenizer=tokenizer,
        n_tokens_target=None,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, return_tensors="pt"
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        # evaluation_strategy="epoch",
        output_dir=str(save_path),
        per_device_eval_batch_size=64,
        predict_with_generate=True,
        fp16=True,
    )

    compute_metrics = functools.partial(_compute_metrics, tokenizer)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for bs in [1, 3, 5]:
        res = trainer.predict(
            dataset,
            num_beams=bs,
            # Faster evaluation
            max_new_tokens=128,
        )
        bleu = res.metrics["test_bleu"]
        chrf = res.metrics["test_chrf"]
        print(f"bleu, bs={bs}", bleu)
        print(f"chrf, bs={bs}", chrf)
        result_file = save_path / f"result_bleu_bs{bs}.txt"
        with result_file.open("w") as fo:
            fo.write(str(bleu))
        result_file = save_path / f"result_chrf_bs{bs}.txt"
        with result_file.open("w") as fo:
            fo.write(str(chrf))


def load_tokenizer(save_path: Path | str) -> transformers.BartTokenizerFast:
    tokenizer = transformers.BartTokenizerFast(tokenizer_file=str(save_path))
    return tokenizer
