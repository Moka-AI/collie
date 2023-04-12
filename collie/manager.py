from __future__ import annotations

import os
import warnings
from typing import ClassVar

from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from collie.types import LanguageModelType
from collie.utils.distributed import fix_file_not_found_in_fsdp


def try_set_fsdp_transformers_cls_name(cls_name: str):
    fsdp_cls_name = os.environ.get('FSDP_TRANSFORMER_CLS_TO_WRAP')
    if fsdp_cls_name == 'auto' or fsdp_cls_name is None:
        os.environ['FSDP_TRANSFORMER_CLS_TO_WRAP'] = cls_name


class BaseModelManager:
    fsdp_transformers_cls_name: ClassVar[str]

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        try_set_fsdp_transformers_cls_name(self.fsdp_transformers_cls_name)

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer

    def build_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        model = model.float()  # type: ignore
        return model


class LLamaManager(BaseModelManager):
    fsdp_transformers_cls_name = 'LlamaDecoderLayer'


class BloomManager(BaseModelManager):
    fsdp_transformers_cls_name = 'BloomBlock'


class ChatGlmManager(BaseModelManager):
    fsdp_transformers_cls_name = 'GLMBlock'

    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.accelerator = Accelerator()

        warnings.filterwarnings(
            'ignore', category=UserWarning, message='.*Creating a tensor from a list of numpy.ndarrays is extremely slow.*'
        )

    def build_tokenizer(self):
        with fix_file_not_found_in_fsdp(self.accelerator, 1):
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        return tokenizer

    def build_model(self):
        with fix_file_not_found_in_fsdp(self.accelerator, 2):
            model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            model = model.float()  # type: ignore
        return model


def get_model_type_from_name(model_name_or_path: str) -> LanguageModelType:
    from transformers import AutoConfig

    mapping = {
        'chatglm': LanguageModelType.chatglm,
        'llama': LanguageModelType.llama,
        'bloom': LanguageModelType.bloom,
    }
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    for k, v in mapping.items():
        if k in config.__class__.__name__.lower():
            return v
    raise ValueError(f'can not infer model type from {model_name_or_path}')


def get_model_manager(model_name_or_path: str, model_type: LanguageModelType | str | None = None) -> BaseModelManager:
    if model_type is None:
        model_type = get_model_type_from_name(model_name_or_path)
    else:
        model_type = LanguageModelType(model_type)

    if model_type is LanguageModelType.chatglm:
        return ChatGlmManager(model_name_or_path)
    elif model_type is LanguageModelType.llama:
        return LLamaManager(model_name_or_path)
    elif model_type is LanguageModelType.bloom:
        return BloomManager(model_name_or_path)
