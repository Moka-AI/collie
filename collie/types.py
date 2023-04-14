from __future__ import annotations

from pathlib import Path
from typing import TypeVar, Sequence, Union
from enum import Enum

from transformers import PreTrainedModel as PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto.auto_factory import _BaseAutoModelClass


T = TypeVar('T')
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
BaseAutoModel = _BaseAutoModelClass
PathOrStr = Union[str, Path]
OneOrMany = Union[T, Sequence[T]]


class LanguageModelType(str, Enum):
    chatglm = 'chatglm'
    llama = 'llama'
    bloom = 'bloom'


class MixedPrecisionType(str, Enum):
    fp16 = 'fp16'
    bf16 = 'bf16'
    no = 'no'
    fp8 = 'fp8'
    auto = 'auto'


class DataType(str, Enum):
    file = 'file'
    hf_dataset = 'hf_dataset'
    auto = 'auto'
