from pathlib import Path
from typing import Any, Literal
import functools
import json

import torch
import transformers  # type: ignore
import tokenizers  # type: ignore
import datasets  # type: ignore
import evaluate  # type: ignore
import numpy as np

from . import config, common


def make_training_args(
    mcfg: config.Mlm,
    *,
    task: bool = False,
) -> transformers.TrainingArguments:
    raise NotImplementedError()  # TODO Handle training/tuning learning rate
    return transformers.TrainingArguments(
        output_dir=mcfg.save_path,
        learning_rate=mcfg.learning_rate,
        per_device_train_batch_size=mcfg.per_device_train_batch_size,
        per_device_eval_batch_size=mcfg.per_device_train_batch_size,
        weight_decay=0.01,
        save_total_limit=mcfg.save_total_limit,
        num_train_epochs=mcfg.n_task_epochs if task else mcfg.n_train_epochs,
        save_steps=mcfg.save_steps,
        logging_steps=mcfg.logging_steps,
    )


def get_tune_data(
    lang_pair: tuple[str, str], dataset_name: str
) -> datasets.DatasetDict:
    raise NotImplementedError()
    pair_str = "-".join(lang_pair)
    rev_pair_str = "-".join(reversed(lang_pair))
    # TODO put in own function
    try:
        raw_dataset = datasets.load_dataset(dataset_name, pair_str)
    except ValueError:
        raw_dataset = datasets.load_dataset(dataset_name, rev_pair_str)

    return raw_dataset["train"].train_test_split(test_size=0.2, seed=0)


@common.check_complete
def init_model(
    *,
    model_cfg: config.Mlm,
    data_path: Path,
) -> None:
    raw_dataset = common.get_raw_dataset(data_path)
    common.make_tokenizer(
        model_cfg=model_cfg,
        raw_dataset=raw_dataset,
    )


@common.check_complete
def train_base_model(*, model_cfg: config.Mlm, data_path: Path | None) -> None:
    hf_model_config = transformers.AlbertConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.hidden_size,
        num_attention_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
    )
    model = transformers.AlbertForMaskedLM(config=hf_model_config)

    if data_path is None:
        model.save_pretrained(str(model_cfg.save_path / "checkpoint-0"))
        return

    raw_dataset = common.get_raw_dataset(data_path)
    common.make_tokenizer(
        model_cfg=model_cfg,
        raw_dataset=raw_dataset,
    )
    tokenizer = common.load_tokenizer(model_cfg)
    dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        save_path=model_cfg.save_path,
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.train_dataset_size,
        cache=True,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
    )

    training_args = make_training_args(model_cfg)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    common.train_try_checkpoint(trainer)


@common.check_complete
def tune_model_mlm(
    *,
    model_cfg: config.Mlm,
    data_path: Path | None,
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)
    tokenizer = common.load_tokenizer(model_cfg)
    model = transformers.AlbertForMaskedLM.from_pretrained(checkpoint_path)
    if model_cfg.finetune_frozen:
        raise NotImplementedError()
        # model.train(False)
        # for p in model.parameters():
        #     p.requires_grad = False
        # for p in model.lm_head.parameters():
        #     p.requires_grad = True
        # model.lm_head.train(True)
    # TODO Need to change the padding IDX
    # model.albert.embeddings.word_embeddings.reset_parameters()
    new_model = transformers.AlbertForMaskedLM(config=model.config)
    fresh_word_embeds = torch.nn.Parameter(
        new_model.albert.embeddings.word_embeddings.weight.detach().clone()
    )
    model.albert.embeddings.word_embeddings.weight = fresh_word_embeds

    if data_path is None:
        model.save_pretrained(str(model_cfg.save_path / "checkpoint-0"))
        return

    raw_dataset = common.get_raw_dataset(data_path)
    train_dataset = common.get_tokenized_dataset(
        model_cfg,
        raw_dataset,
        save_path=model_cfg.tokenizer_path.parents[0],
        tokenizer=tokenizer,
        n_tokens_target=model_cfg.tune_dataset_size,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
    )
    training_args = make_training_args(model_cfg)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    common.train_try_checkpoint(trainer)


def get_task_dataset(
    model_cfg: config.Mlm,
    *,
    tokenizer: transformers.PreTrainedTokenizerFast,
    phase: Literal["train", "test"],
) -> tuple[datasets.arrow_dataset.Dataset, list[str]]:
    raw_dataset = datasets.load_dataset(
        model_cfg.downstream_dataset, name=model_cfg.downstream_subset
    )

    label_list = get_label_list(raw_dataset["test"], model_cfg.label_feat)
    label_map = {v: i for i, v in enumerate(label_list)}

    raw_dataset = raw_dataset[phase]
    if phase == "train":
        raw_dataset = raw_dataset.select(range(model_cfg.max_task_examples))

    string_labels = isinstance(raw_dataset[model_cfg.label_feat][0][0], str)

    def tokenize_and_align_labels(examples: Any) -> dict:
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[model_cfg.label_feat]):
            if string_labels:
                label = [label_map.get(l.split(":")[0], -100) for l in label]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return raw_dataset.map(tokenize_and_align_labels, batched=True), label_list


def compute_metrics(eval_preds: tuple, *, label_list: list[str]) -> dict[str, Any]:
    seqeval = evaluate.load("seqeval")

    preds, labels = eval_preds
    preds = np.argmax(preds, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(ps, ls) if l != -100]
        for ps, ls in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(ps, ls) if l != -100]
        for ps, ls in zip(preds, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_label_list(raw_dataset: Any, label_feat: str) -> list[str]:
    if isinstance(raw_dataset[label_feat][0][0], str):
        # TODO Necessary for deprel
        label_list = list(
            set(x.split(":")[0] for y in raw_dataset[label_feat] for x in y)
        )
        label_list = sorted(label_list)
    else:
        label_list = raw_dataset.features[label_feat].feature.names

    # if label_feat not in ["ner_tags", "chunk_tags"]:
    #     label_list = ["B-" + l for l in label_list]

    return label_list


@common.check_complete
def tune_model_task(
    *,
    model_cfg: config.Mlm,
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)
    tokenizer = common.load_tokenizer(model_cfg)

    tokenized_dataset, label_list = get_task_dataset(
        model_cfg,
        tokenizer=tokenizer,
        phase="train",
    )

    data_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    model = transformers.AlbertForTokenClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(label_list),
        # id2label=id2label,
        # label2id=label2id,
    )
    training_args = make_training_args(model_cfg, task=True)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )
    common.train_try_checkpoint(trainer)


@common.check_complete
def compute_score(
    *,
    model_cfg: config.Mlm,
) -> None:
    checkpoint_path = common.get_checkpoint(model_cfg.load_path)
    model = transformers.AlbertForTokenClassification.from_pretrained(checkpoint_path)
    model.train(False)
    tokenizer = common.load_tokenizer(model_cfg)
    tokenized_dataset, label_list = get_task_dataset(
        model_cfg,
        tokenizer=tokenizer,
        phase="test",
    )

    data_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=tokenizer,
    )
    model = transformers.AlbertForTokenClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(label_list),
        # id2label=id2label,
        # label2id=label2id,
    )
    training_args = make_training_args(model_cfg, task=True)

    _compute_metrics = functools.partial(compute_metrics, label_list=label_list)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    res = trainer.predict(
        tokenized_dataset,
    )
    f1 = res.metrics["test_f1"]
    json.dump(res.metrics, (model_cfg.save_path / "all_results.json").open("w"))
    print(f"f1", f1)
    result_file = model_cfg.save_path / f"result.txt"
    with result_file.open("w") as fo:
        fo.write(str(f1))
