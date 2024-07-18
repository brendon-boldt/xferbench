import pytest
import pydantic

from xferbench.model import config


def assert_field_equal(
    x: type[pydantic.BaseModel], y: type[pydantic.BaseModel], field: str
) -> None:
    assert x.model_fields[field].default == y.model_fields[field].default


def test_multiple_config_inheritance() -> None:
    assert_field_equal(config.ClmTest, config.ModelTest, "train_dataset_size")
    assert_field_equal(config.ClmTest, config.Clm, "tokenizer_class")

    assert_field_equal(config.MtTest, config.ModelTest, "train_dataset_size")
    assert_field_equal(config.MtTest, config.Mt, "tokenizer_class")
