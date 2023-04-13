from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from collie.utils.trainer import LossTracker, CollieTqdmProgressBar


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
        step = 0
        self.progress_bar = CollieTqdmProgressBar(self.epochs, len(self.train_dataloader))

        for current_epoch in range(1, self.epochs + 1):
            self.model.train()

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

                self.progress_bar.update()
                step += 1
                if batch_index % self.log_interval == 0:
                    self.log_metrics({'loss': self.train_loss_tracker.loss}, step=step)

            train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss}, 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_loss_tracker.end_epoch()
            self.progress_bar.end_epoch()

            if self.validation_dataloader:
                validation_loss = evaluate(self.model, self.validation_dataloader, self.validation_loss_tracker)
                validation_metrics = self.add_prefix({'loss': validation_loss}, 'validation')
                self.accelerator.log(validation_metrics, step=current_epoch)

            if self.save_on_epoch_end:
                self.accelerator.save_state()

            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)

        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


def evaluate(model: torch.nn.Module, dataloader: DataLoader, loss_tracker: LossTracker | None = None):
    model = model.eval()
    loss_tracker = loss_tracker or LossTracker()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(**batch)
            loss_tracker.update(batch_output['loss'])
    loss = loss_tracker.loss
    loss_tracker.end_epoch()
    return loss
