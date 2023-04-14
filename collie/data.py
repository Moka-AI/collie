from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any, List, Sequence, cast

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_dataset, load_from_disk

from collie.types import Tokenizer, PathOrStr
from collie.utils.io import load_dataset_from_file
from collie.utils.tokenizer import infer_num_end_tokens

IGNORE_INDEX = -100


class PreTokenizedCollator:
    def __init__(self, tokenizer: Tokenizer):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                warnings.warn('pad_token_id is None, will use eos token as pad_token')
            else:
                raise ValueError('pad_token_id is None, eos_token is None, can not pad')

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        self.pad_token_id = tokenizer.pad_token_id or 0
        self.padding_side = tokenizer.padding_side

    def __call__(self, records: Sequence[dict[str, Any]]):
        instruction_length_list = [record.pop('num_instruction_ids') for record in records]
        merged_records = {}
        for key in records[0].keys():
            values = [record[key] for record in records]
            merged_records[key] = values
        batch_encodes = self.tokenizer.pad(merged_records, return_tensors='pt')  # type: ignore

        input_ids = batch_encodes['input_ids']
        input_ids = cast(torch.Tensor, input_ids)
        labels = self.create_labels(input_ids, instruction_length_list, self.padding_side, self.pad_token_id)
        batch_encodes['labels'] = labels

        return batch_encodes

    @staticmethod
    def create_labels(input_ids: torch.Tensor, instruction_length: list[int], padding_side: str, pad_token_id: int):
        labels = input_ids.clone()
        if padding_side == 'right':
            for i, length in enumerate(instruction_length):
                labels[i, :length] = IGNORE_INDEX
            labels[labels == pad_token_id] = IGNORE_INDEX
        elif padding_side == 'left':
            pad_length = (labels == pad_token_id).sum(dim=-1).tolist()
            instruction_length = [i + j for i, j in zip(instruction_length, pad_length)]
            for i, length in enumerate(instruction_length):
                labels[i, :length] = IGNORE_INDEX
            labels[labels == pad_token_id] = IGNORE_INDEX
        return labels


class InstructionDataset(Dataset):
    def __init__(self, processed_dataset: HFDataset):
        self.data = processed_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_name_or_path(
        cls,
        name_or_path: str,
        tokenizer: Tokenizer,
        max_length: int | None = None,
        num_proc: int | None = None,
        num_end_tokens: int | None = None,
    ):
        if Path(name_or_path).is_file():
            return cls.from_file(
                name_or_path, tokenizer, max_length=max_length, num_proc=num_proc, num_end_tokens=num_end_tokens
            )
        else:
            if Path(name_or_path).is_dir():
                dataset = load_from_disk(name_or_path)
            else:
                dataset = load_dataset(name_or_path)

            if isinstance(dataset, dict):
                dataset = dataset['train']
            dataset = cast(HFDataset, dataset)
            return cls.from_raw_dataset(
                dataset, tokenizer, max_length=max_length, num_proc=num_proc, num_end_tokens=num_end_tokens
            )

    @classmethod
    def from_file(
        cls,
        file_path: PathOrStr,
        tokenizer: Tokenizer,
        file_type: str = 'auto',
        max_length: int | None = None,
        num_proc: int | None = None,
        num_end_tokens: int | None = None,
    ):
        dataset = load_dataset_from_file(file_path, file_type)
        return cls.from_raw_dataset(dataset, tokenizer, max_length, num_proc, num_end_tokens)

    @classmethod
    def from_raw_dataset(
        cls,
        dataset: HFDataset,
        tokenizer: Tokenizer,
        max_length: int | None = None,
        num_proc: int | None = None,
        num_end_tokens: int | None = None,
    ):
        assert set(dataset[0].keys()) == {'output', 'instruction'}

        if num_proc is not None and num_proc > 1:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        max_length = max_length or tokenizer.model_max_length
        print(f'max_length: {max_length}')

        if num_end_tokens is None:
            num_end_tokens = infer_num_end_tokens(tokenizer)

        def _process_instruction_record(record):
            return {
                'model_input': record['instruction'] + record['output'],
                'instruction': record['instruction'],
            }

        dataset = dataset.map(_process_instruction_record, num_proc=num_proc, remove_columns=['output', 'instruction'])

        def tokenize_record(record):
            encodes = tokenizer(record['model_input'], max_length=max_length, truncation=True, return_token_type_ids=False)
            instruction_ids = tokenizer(record['instruction'], max_length=max_length, truncation=True)['input_ids']
            instruction_ids = cast(List[int], instruction_ids)
            num_instruction_ids = min(len(instruction_ids), max_length) - num_end_tokens
            encodes['num_instruction_ids'] = num_instruction_ids  # type: ignore
            return encodes

        dataset = dataset.map(tokenize_record, num_proc=num_proc, remove_columns=['model_input', 'instruction'])
        return cls(dataset)


class DebugDataset(Dataset):
    def __init__(self, vocab_size: int = 100, num_samples: int = 2000, max_sequence_length: int = 1024):
        self.vocab_size = vocab_size
        self.samples = [
            torch.randint(low=0, high=vocab_size, size=(random.randint(10, max_sequence_length),)) for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            'input_ids': self.samples[index],
            'labels': self.samples[index],
        }
