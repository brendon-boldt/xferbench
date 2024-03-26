# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from tqdm import tqdm  # type: ignore

from transformers import (  # type: ignore
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BartConfig,
    BatchEncoding,
    FlaxBartForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.bart.modeling_bart import shift_tokens_right  # type: ignore


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling. The code is largely copied from
    `<https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L223>`__.
    For more information on how BART denoising language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.13461.pdf>`__
    or the `official code for preprocessing <https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    # Not going to do permutation because we don't have a definition of
    # sentences for non-human languages.
    # permute_sentence_ratio: float = 1.0
    permute_sentence_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # Below is slow
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        batch["labels"] = batch["input_ids"].copy()

        # EDIT: This script is written for FLAX but we want to use PyTorch.
        # shift_tokens_right expects PyTorch tensors, though, so we convert
        # them to call this function (so we don't have to modify a second
        # file), and then convert them back to numpy arrays.
        batch["labels"] = torch.from_numpy(batch["labels"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        batch["decoder_input_ids"] = batch["decoder_input_ids"].numpy()
        # END EDIT

        # permuting sentences
        do_permute = False
        # if self.permute_sentence_ratio > 0.0:
        #     batch["input_ids"] = self.permute_sentences(batch["input_ids"])
        #     do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.span_mask_tokens(
                batch["input_ids"], batch["labels"], do_permute
            )

        # ignore pad tokens
        # Below is slow
        batch["attention_mask"] = (
            batch["input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        batch["decoder_attention_mask"] = (
            batch["decoder_input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v)
        return batch

    def span_mask_tokens(self, input_ids, labels, do_permute) -> Any:
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        _special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        _special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(_special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(_special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = (
            ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        )
        num_tokens_to_mask = int(
            math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio)
        )
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(
            lam=self.poisson_lambda, size=(num_tokens_to_mask,)
        )
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [
                    span_lengths,
                    np.random.poisson(
                        lam=self.poisson_lambda, size=(num_tokens_to_mask,)
                    ),
                ]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = (
            np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        )
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[
            : span_lengths.shape[0]
        ]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels


def generate_batch_splits(
    samples_idx: np.ndarray, batch_size: int, drop_last=True
) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned.
    """
    num_samples = len(samples_idx)
    _samples_idx: Any
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        _samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        _samples_idx = np.array_split(samples_idx, sections_split)
    return _samples_idx
