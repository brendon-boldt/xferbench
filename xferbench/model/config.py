from typing import Any, cast
from pathlib import Path
import sys

import transformers  # type: ignore
import pydantic

target_languages = [
    "da",
    "eu",
    "fa",
    "fi",
    "he",
    "id",
    "ja",
    "kk",
    "ro",
    "ur",
]

TokenizerClass = type[transformers.PreTrainedTokenizerFast]


class Model(pydantic.BaseModel):
    save_path: Path
    tokenizer_path: Path
    load_path: Path | None = None

    train_learning_rate: float = 1e-4
    # Untested; previously was 2e-5
    tune_learning_rate: float = 1e-4
    # Borrowing from ALBERT's default
    vocab_size: int = 30000
    context_length: int = 1 << 8
    n_train_epochs: int = 5
    n_tune_epochs: int = 10
    per_device_train_batch_size: int = 1 << 5
    # Fraction of total training
    save_steps: float = 1 / 20
    save_total_limit: int = 1
    logging_steps: float = 1 / 100
    finetune_frozen: bool = False
    tokenizer_class: TokenizerClass

    # Measured in tokens
    train_dataset_size: int = 15_000_000
    tune_dataset_size: int = 2_000_000

    def init_dirs(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path.parents[0].mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.init_dirs()
        with (self.save_path / "model.config.json").open("w") as fo:
            fo.write(self.model_dump_json(indent=4))

    @pydantic.field_serializer("tokenizer_class")
    def serialize_class(self, tkc: TokenizerClass, _info) -> str:
        return f"{tkc.__module__}.{tkc.__name__}"

    @pydantic.field_validator("tokenizer_class", mode="before")
    @classmethod
    def validate_class(cls, v: str | TokenizerClass, _info) -> TokenizerClass:
        if isinstance(v, str):
            return eval(v)
        else:
            return v


class Clm(Model):
    # This uses Gpt2 naming conventions
    n_head: int = 6
    n_layer: int = 6
    tokenizer_class: TokenizerClass = transformers.GPT2TokenizerFast


class ModelTest(pydantic.BaseModel):
    vocab_size: int = 1 << 10
    context_length: int = 1 << 6
    n_train_epochs: int = 1
    n_tune_epochs: int = 1
    per_device_train_batch_size: int = 4
    save_steps: float = 1
    save_total_limit: int = 2
    logging_steps: float = 1
    train_dataset_size: int = 20
    tune_dataset_size: int = 20


class ClmTest(ModelTest, Clm):
    n_head: int = 2
    n_layer: int = 2


class Mlm(Model):
    # Uses Albert naming conventions
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    per_device_train_batch_size: int = 1 << 4
    tokenizer_class: TokenizerClass = transformers.AlbertTokenizerFast

    n_task_epochs: int = 10
    # Check this
    tune_dataset_size: int = 100_000
    max_task_examples: int = 1000
    downstream_dataset: str = "conll2003"
    label_feat: str = "DEFAULT"
    downstream_subset: str | None = None


class MlmLL(Mlm):
    tune_dataset_size: int = 2_000_000
    max_task_examples: int = 14_000


class MlmLS(Mlm):
    tune_dataset_size: int = 2_000_000
    max_task_examples: int = 1000


class MlmXsL(Mlm):
    tune_dataset_size: int = 1000
    max_task_examples: int = 14_000


class MlmXsXs(Mlm):
    tune_dataset_size: int = 1000
    max_task_examples: int = 100


this_mod = sys.modules[__name__]

for base_model in [MlmLL, MlmLS, MlmXsXs, MlmXsL]:
    for task in ["ner", "chunk", "pos"]:
        new_model = pydantic.create_model(
            base_model.__name__ + "_" + task,
            __base__=base_model,
            label_feat=(str, f"{task}_tags"),
        )
        setattr(this_mod, new_model.__name__, new_model)

for base_model in [MlmLL, MlmLS, MlmXsXs, MlmXsL]:
    task_examples_kwarg = (
        {"max_task_examples": (int, 12_000)} if base_model in [MlmXsL, MlmLL] else {}
    )
    new_model = pydantic.create_model(
        base_model.__name__ + "_deprel",
        __base__=base_model,
        label_feat=(str, "deprel"),
        task_dataset=(str, "universal_dependencies"),
        task_dataset_subset=(str, "en_ewt"),
        **cast(Any, task_examples_kwarg),
    )
    setattr(this_mod, new_model.__name__, new_model)


class Mt(Model):
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    d_model: int = 512
    # n_train_epochs: int = 3
    n_tune_epochs: int = 3
    max_position_embeddings: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    tokenizer_class: TokenizerClass = transformers.BartTokenizerFast
    task_dataset: str = "wmt14"
    tune_learning_rate: float = 2e-4


class MtWmt(Mt):
    task_dataset: str = "wmt14"
    task_dataset_subset: str = "en-fr"
    train_dataset_size: int = 100_000_000
    tune_dataset_size: int = 50_000_000


class MtMicroTune(MtWmt):
    tune_dataset_size: int = 2_000_000


class MtLowLr(MtWmt):
    tune_learning_rate: float = 2e-5


class MtFreeze(MtWmt):
    finetune_frozen: bool = True


class MtLL(MtWmt):
    tune_learning_rate: float = 2e-5
    tune_dataset_size: int = 10_000_000


class MtTest(ModelTest, Mt):
    pass
