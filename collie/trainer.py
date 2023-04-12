from __future__ import annotations

from typing import Any

import torch
import tqdm
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from collie.utils.trainer import LossTracker, DummyProgressBar


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None,
        accelerator: Accelerator,
        epochs: int,
        lr_scheduler: LRScheduler | None = None,
        log_interval: int = 50,
        save_on_epoch_end: bool = True,
        epoch_end_callbacks: list[Any] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        self.epoch_end_callbacks = epoch_end_callbacks or []

    def train(self):
        num_cumulate_batch = 0
        for current_epoch in range(1, self.epochs + 1):

            self.set_progress_bar()
            self.model = self.model.train()
            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)

                self.progress_bar.update(1)
                num_cumulate_batch += 1

                if batch_index % self.log_interval == 0:
                    self.accelerator.log({'loss': self.train_loss_tracker.loss}, step=num_cumulate_batch)
                    self.set_progress_bar_description(current_epoch, {'loss': self.train_loss_tracker.loss})

            train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss}, 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_loss_tracker.end_epoch()
            self.progress_bar.close()

            if self.validation_dataloader:
                self.model = self.model.eval()
                for batch in self.validation_dataloader:
                    with torch.inference_mode():
                        batch_output = self.model(**batch)
                        self.validation_loss_tracker.update(batch_output['loss'])

                validation_metrics = self.add_prefix({'loss': self.validation_loss_tracker.loss}, 'validation')
                self.accelerator.log(validation_metrics, step=current_epoch)
                self.validation_loss_tracker.end_epoch()

            if self.save_on_epoch_end:
                self.accelerator.save_state()

            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)

        self.accelerator.end_training()

    def set_progress_bar(self) -> None:
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(total=len(self.train_dataloader))
        else:
            self.progress_bar = DummyProgressBar(total=len(self.train_dataloader))

    def set_progress_bar_description(self, current_epoch: int, metrics: dict[str, float]) -> None:
        description = f'Epoch {current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            description += f' - {name}: {score:.4f}'
        self.progress_bar.set_description(description)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}
