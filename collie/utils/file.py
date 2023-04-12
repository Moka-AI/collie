from typing import cast
from pathlib import Path

from datasets import load_dataset, DatasetDict

from collie.types import PathOrStr


def convert_suffix_to_type(suffix: str):
    suffix_type_mapping = {
        'jsonl': 'json',
        'txt': 'text',
        'tsv': 'csv',
    }
    suffix = suffix_type_mapping.get(suffix, suffix)
    return suffix


def load_dataset_from_file(file_path: PathOrStr, file_type: str = 'auto', **kwargs):
    file_path = Path(file_path)

    dataset_kwargs = kwargs
    if file_type == 'auto':
        suffix = file_path.suffix.lstrip('.')
        file_type = convert_suffix_to_type(suffix)
        if file_type not in ('csv', 'json'):
            raise ValueError(f'file_type {file_type} is not supported')
        if suffix == 'tsv':
            dataset_kwargs['sep'] = '\t'

    dataset_dict = load_dataset(file_type, data_files={'train': str(file_path)}, **dataset_kwargs)
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset = dataset_dict['train']
    return dataset
