from __future__ import annotations
import os

from pathlib import Path
from typing import Optional

import typer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, PrecisionType
from transformers import get_cosine_schedule_with_warmup

from collie.trainer import Trainer
from collie.types import LanguageModelType
from collie.manager import get_model_manager, get_model_type_from_name
from collie.data import InstructionDataset, PreTokenizedCollator
from collie.utils.io import save_to_disk


def create_adamw_optimizer(model: torch.nn.Module, lr: float, weight_decay=1e-3):
    parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'layernorm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def main(
    model_name_or_path: str,
    train_file: str,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 4,
    num_workers: int = 4,
    max_length: Optional[int] = 1024,
    seed: int = 42,
    lr: float = 2e-5,
    weight_decay: float = 1e-3,
    mixed_precision: PrecisionType = PrecisionType.BF16,
    save_precision: Optional[PrecisionType] = None,
    gradient_accumulation_steps: int = 1,
    model_type: Optional[LanguageModelType] = None,
    save_on_epoch_end: bool = False,
    num_max_checkpoints: int = 1,
    num_warmup_steps: float = 0.05,
    use_tensorboard: bool = False,
):
    # Prepare Accelerator
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

    if model_type is None:
        model_type = get_model_type_from_name(model_name_or_path)

    if save_precision is None:
        save_precision = mixed_precision

    output_dir = output_dir or Path('experiments') / f'{model_type}-fft'
    project_config = ProjectConfiguration(
        project_dir=str(output_dir), automatic_checkpoint_naming=True, total_limit=num_max_checkpoints
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=['tensorboard'] if use_tensorboard else None,
    )
    accelerator.init_trackers(f'{model_type}-fft')

    set_seed(seed)
    accelerator.print(f'Start with seed: {seed}')
    accelerator.print(f'Output dir: {output_dir}')

    model_manager = get_model_manager(model_name_or_path, model_type)

    # DataLoader
    tokenizer = model_manager.build_tokenizer()
    dataset = InstructionDataset.from_file(train_file, tokenizer, max_length=max_length, num_proc=num_workers)
    data_collator = PreTokenizedCollator(tokenizer)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    # Model
    model = model_manager.build_model()
    model = accelerator.prepare(model)

    # Optimizer
    optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)

    # LRScheduler
    total_steps = len(train_dataloader) * epochs
    if num_warmup_steps < 1:
        num_warmup_steps = int(num_warmup_steps * total_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_warmup_steps),
        num_training_steps=total_steps,
    )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=None,
        accelerator=accelerator,
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        log_interval=10,
        save_on_epoch_end=save_on_epoch_end,
    )
    accelerator.print(f'Start training for {epochs} epochs')
    trainer.train()

    accelerator.wait_for_everyone()
    accelerator.print('Training finished')

    accelerator.print('Saving model')
    unwrapped_model = accelerator.unwrap_model(model)
    save_to_disk(output_dir / 'model', model=unwrapped_model, tokenizer=tokenizer, save_precision=save_precision)


if __name__ == '__main__':
    typer.run(main)
