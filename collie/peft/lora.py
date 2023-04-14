from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class LoraDict(nn.ModuleDict):

    def detach(self):
        for lora_module in self.values():
            lora_module = cast(LoraLinear, lora_module)
            lora_module.detach()


class LoraLinear(nn.Module):

    def __init__(
        self, 
        in_features: int,
        out_features: int,
        rank: int= 8,
        alpha: float=1.0,
        dropout: float=0,
        bias: bool=False, 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.bias = bias

        self.low_rank_linear = nn.Sequential(
            nn.Linear(in_features, rank, bias=bias),
            nn.Linear(rank, out_features, bias=bias),
        )
        
        self.attached: dict[str, Any] = {
            'linear': None,
            'forward': None, 
        }

    def attach(self, linear_module: nn.Linear, replace_forward: bool=True):
        self.attached['linear'] = linear_module
        if replace_forward:
            self.attached['forward'] = linear_module.forward
            linear_module.forward = self.forward

    def detach(self, merge_weights: bool=False):
        frozen_linear = self.attached['linear']
        frozen_linear_forward = self.attached['forward']

        if frozen_linear is None:
            raise RuntimeError("You must attach a linear module to this LoRA module before detaching it")
        if frozen_linear_forward is not None:
            frozen_linear.forward = frozen_linear_forward
        if merge_weights:
            self.merge_weights(frozen_linear)
        self.attached['linear'] = None
        self.attached['forward'] = None
    
    def merge_weights(self, linear_module: nn.Linear):
        linear_module.weight.data += self.low_rank_linear(torch.eye(self.in_features)).t() * (self.alpha / self.rank)

    @classmethod
    def from_linear(cls, linear_module: nn.Linear, rank: int=8, alpha: float=1.0, dropout: float=0, bias: bool=False, auto_attach: bool=True):
        in_features = linear_module.in_features
        out_features = linear_module.out_features
        lora = cls(in_features, out_features, rank=rank, alpha=alpha, dropout=dropout, bias=bias)
        if auto_attach:
            lora.attach(linear_module)
        return lora

    def forward(self, input: torch.Tensor):
        frozen_linear = self.attached['linear']
        if frozen_linear is None:
            raise RuntimeError("You must attach a linear module to this LoRA module before calling forward")

        linear_output =  nn.functional.linear(input, frozen_linear.weight, frozen_linear.bias)
        low_rank_output = self.low_rank_linear(self.dropout(input)) * (self.alpha / self.rank)
        return linear_output + low_rank_output


def create_lora_model(base_model: nn.Module):
    lora_modules = LoraDict()
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoraLinear.from_linear(module)
            name = name.replace('.', '/')
            lora_modules[name] = lora
    return base_model, lora_modules


def load_lora_model_from_pretrained(base_model: nn.Module, lora_modules_state_dict: dict[str, Any], merge: bool=False):
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            name = name.replace('.', '/')
            if name not in lora_modules_state_dict:
                continue

            if merge:
                lora_linear = LoraLinear.from_linear(module, auto_attach=False)
                lora_linear.load_state_dict(lora_modules_state_dict[name])
                lora_linear.merge_weights(module)
            else:
                lora_linear = LoraLinear.from_linear(module, auto_attach=True)
                lora_linear.load_state_dict(lora_modules_state_dict[name])

    return base_model
