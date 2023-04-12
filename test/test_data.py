import json

import torch
import pytest

from collie.manager import get_model_manager
from collie.data import InstructionDataset, PreTokenizedCollator

from test import FIXTURES_DIR, MODELS_DIR


@pytest.mark.parametrize('model_name', ['mini-chatglm', 'mini-llama', 'mini-bloom'])
def test_data_pipeline(model_name: str):
    model_path = MODELS_DIR / model_name
    model_manager = get_model_manager(str(model_path))
    tokenizer = model_manager.build_tokenizer()
    jsonl_file = FIXTURES_DIR / 'debug_instruction.jsonl'
    dataset = InstructionDataset.from_file(jsonl_file, tokenizer=tokenizer, max_length=128)
    instruction_records = []
    with open(jsonl_file) as f:
        for line in f:
            if line.strip():
                instruction_records.append(json.loads(line))
    collator = PreTokenizedCollator(tokenizer)

    encodes = collator([dataset[i] for i in range(len(dataset))])
    labels = encodes['labels']

    assert isinstance(labels, torch.Tensor)
    assert labels.shape[0] == len(instruction_records)
    special_id = tokenizer.all_special_ids[0]
    for sample_labels, record in zip(labels, instruction_records):
        sample_labels[sample_labels == -100] = special_id
        text = tokenizer.decode(sample_labels, skip_special_tokens=True)
        assert text == record['output']
