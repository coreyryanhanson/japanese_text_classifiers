"""A module for PyTorch model and training classes."""

from contextlib import contextmanager
from typing import Optional, Union

import pandas as pd
import torch


class TrainerGeneric:
    """A base class for the management of PyTorch training loops.

    Args:
        model (object): A class that inherets from the PyTorch module base
            class.
        optimizer (object): A PyTorch optimizer
        criterion (object): A PyTorch  loss function class.
        dataloaders (dict[str, object]): A dictionary of data loaders with keys
            for "train", "val", and "test".
        train_metrics (Optional[list[str]], optional): A list of column names
            that will be tracked in the training results DataFrames. Defaults
            to None.
        scheduler (Optional[object], optional): A pytorch scheduler object.
            Defaults to None.
    """
    def __init__(self,
                 model: object,
                 optimizer: object,
                 criterion: object,
                 dataloaders: dict[str, object],
                 train_metrics: Optional[list[str]] = None,
                 scheduler: Optional[object] = None
                 ) -> None:
        self.epoch_col = "epoch"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.dataloaders: dict[str, object] = {}
        self.batch_size: dict[str, int] = {}
        self.train_metrics_cols = self._set_train_metrics_cols(train_metrics)
        self.current_results = None
        self.complete_results = None
        self._last_learning_rates: list[float] = []
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.set_dataloaders(dataloaders)

    @contextmanager
    def _device_context(self, instance: object) -> object:
        yield instance.to(self.device)
        instance.to("cpu")

    def _backpropogate(self, loss: object, zero_grad: bool = True) -> None:
        if zero_grad:
            self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _document_metrics(self, epoch: int, loss: list[float]) -> pd.DataFrame:
        index = pd.Index([epoch], name=self.epoch_col)
        output = pd.DataFrame([loss + self._last_learning_rates],
                              columns=self.train_metrics_cols,
                              index=index, dtype=float)
        print(output)
        return output

    def _set_train_metrics_cols(self, train_metrics: list[str]) -> list[str]:
        opt_params = self.optimizer.param_groups
        if len(opt_params) == 1:
            learning_rate_cols = ["lr"]
        else:
            learning_rate_cols = [f"lr{i}" for i in range(len(opt_params))]
        return train_metrics + learning_rate_cols if train_metrics else learning_rate_cols

    def _update_results(self,
                        results_list: list[pd.DataFrame]
                        ) -> pd.DataFrame:
        self.current_results = pd.concat(results_list, axis=0)
        if self.complete_results is None:
            self.complete_results = self.current_results
        else:
            self.current_results.index = self.current_results.index + self.complete_results.index[-1]
            self.complete_results = pd.concat([self.complete_results,
                                               self.current_results], axis=0)

    def _step(self, process: str) -> list[float]:
        """Override this function"""
        print(f"empty {process}")
        return []

    def _capture_lr(self) -> None:
        optimizer_group_count = len(self.optimizer.param_groups)
        self._last_learning_rates = [self.optimizer.param_groups[i]["lr"]
                                     for i in range(optimizer_group_count)]

    def set_dataloaders(self, dataloaders: dict[str, object]) -> None:
        """Stores a set of dataloader objects within a dictionary and makes
        an additional dict to store batch sizes.

        Args:
            dataloaders (dict[str, object]): A dictionary of data loaders with keys
                for "train", "val", and "test".
        """
        self.dataloaders = dataloaders
        self.batch_size = {key: value.batch_size for key, value in dataloaders.items()}

    def override_lr(self,
                    learning_rate: float,
                    override_indices: Optional[Union[int, list[int]]] = None
                    ) -> None:
        """Overrides the learning rate within the object's stored optimizer.

        Args:
            learning_rate (float): The new learning rate.
            override_indices (Optional[Union[int, list[int]]], optional):
                If there are multiple param groups on the optimizer, this will
                specify which to override. If left empty, all will be updated
                to the provided value. Defaults to None.
        """
        if override_indices is None:
            override_indices = range(len(self.optimizer.param_groups))
        elif isinstance(override_indices, int):
            override_indices = [override_indices]
        for i in override_indices:
            self.optimizer.param_groups[i]["lr"] = learning_rate

    def train(self, epochs: int) -> None:
        """Initializes the training loop while logging results of each epoch
        to DataFrames stored on the instance.

        Args:
            epochs (int): The number of epochs to train for. Training can be
                resumed after stopping.
        """
        metrics = []
        with self._device_context(self.model):
            for epoch in range(1, epochs + 1):
                train_loss = self._step("train")
                self._capture_lr()
                if self.scheduler is not None:
                    self.scheduler.step()
                val_loss = self._step("val")
                metrics.append(self._document_metrics(epoch,
                                                      train_loss + val_loss))
        self._update_results(metrics)
