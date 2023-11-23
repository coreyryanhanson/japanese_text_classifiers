"""A module for PyTorch model and training classes."""

from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn


class TrainerGeneric:
    """A base class for the management of PyTorch training loops.

    Args:
        model (object): A class that inherets from the PyTorch module base
            class.
        optimizer (object): A PyTorch optimizer
        criterion (object): A PyTorch  loss function class.
        dataloaders (dict[str, object]): A dictionary of data loaders with keys
            for "train", "val", and "test".
        scheduler (Optional[object], optional): A pytorch scheduler object.
            Defaults to None.
    """
    def __init__(self,
                 model: object,
                 optimizer: object,
                 criterion: object,
                 dataloaders: dict[str, object],
                 scheduler: Optional[object] = None
                 ) -> None:
        self.epoch_col = "epoch"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.dataloaders: dict[str, object] = {}
        self.batch_size: dict[str, int] = {}
        self.metrics_cols = []
        self.learning_rate_cols = self._set_learning_rate_cols()
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

    def _force_list(self, var):
        if not isinstance(var, list):
            return [var]
        return var

    def _backpropogate(self, loss: object, zero_grad: bool = True) -> None:
        if zero_grad:
            self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _document_metrics(self,
                          epoch: int,
                          loss: Union[list[list[float]], list[float]],
                          counts: Union[list[int], int],
                          process: Union[list[str], str]
                          ) -> pd.DataFrame:
        # Allowing both a single process and lists of several.
        if not isinstance(loss[0], list):
            loss = [loss]
        counts, process = [self._force_list(val) for val in [counts, process]]
        column_names, loss_expanded = [], []
        for i in range(len(process)):
            for j in range(len(self.metrics_cols)):
                column_names.append(process[i] + "_" + self.metrics_cols[j])
                loss_expanded.append(loss[i][j])
            column_names.append("n_" + process[i])
            loss_expanded.append(counts[i])
        index = pd.Index([epoch], name=self.epoch_col)
        output = pd.DataFrame([loss_expanded + self._last_learning_rates],
                              columns=column_names + self.learning_rate_cols,
                              index=index, dtype=float)
        print(output)
        return output

    def _get_last_epoch(self) -> int:
        if self.complete_results is None:
            self._capture_lr()
            return 0
        return self.complete_results.index[-1]

    def _set_learning_rate_cols(self) -> list[str]:
        opt_params = self.optimizer.param_groups
        if len(opt_params) == 1:
            return ["lr"]
        else:
            return [f"lr{i}" for i in range(len(opt_params))]

    def _combine_results_rows(self,
                              results_list: list[pd.DataFrame]
                              ) -> pd.DataFrame:
        df = pd.concat(results_list, axis=0)
        for column in df.columns[df.columns.str.startswith("n_")].values:
            df[column] = df[column].astype(int)
        return df

    def _update_results(self, results_list: list[pd.DataFrame]) -> None:
        self.current_results = self._combine_results_rows(results_list)
        if self.complete_results is None:
            self.complete_results = self.current_results
        else:
            self.current_results.index = (self.current_results.index +
                                          self._get_last_epoch())
            self.complete_results = pd.concat([self.complete_results,
                                               self.current_results], axis=0)

    def _process_batch(self,
                       batch: torch.Tensor,
                       labels: torch.Tensor,
                       is_train: bool):
        """Captures loss and predicted label values with optional
        backpropagation if is_train is enabled.
        """
        batch = batch.to(self.device)
        labels = labels.to(self.device, copy=True)
        outputs = self.model(batch)[0]
        loss = self.criterion(outputs, labels)
        if is_train:
            self._backpropogate(loss, zero_grad=True)
        _, predicted = torch.max(outputs, 1)
        return loss.item(), predicted.to("cpu")

    def _step(self, process: str) -> tuple[list[float], int]:
        """Override this function"""
        print(f"empty {process}")
        return [], 0

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
        train_key = "train"
        val_key = "val"
        with self._device_context(self.model):
            for epoch in range(1, epochs + 1):
                train_loss, n_train = self._step(train_key)
                self._capture_lr()
                if self.scheduler is not None:
                    self.scheduler.step()
                val_loss, n_val = self._step(val_key)
                metrics.append(self._document_metrics(epoch,
                                                      [train_loss, val_loss],
                                                      [n_train, n_val],
                                                      [train_key, val_key]))
        self._update_results(metrics)

    def test(self):
        with self._device_context(self.model):
            key = "test"
            loss, n = self._step(key)
            return self._document_metrics(self._get_last_epoch(), loss, n, key)


class CharacterTrainer(TrainerGeneric):
    def __init__(self, model, optimizer, criterion, dataloaders, scheduler=None, balance_acc=False):
        super().__init__(model, optimizer, criterion, dataloaders, scheduler)
        self.metrics_cols = ["loss", "acc"]
        self.accuracy_func = (balanced_accuracy_score if balance_acc
                              else accuracy_score)

    def _calc_accuracy(self, labels, predictions):
        return self.accuracy_func(np.concatenate(labels),
                                  np.concatenate(predictions))

    def _step(self, process):

        epoch_labels, epoch_predictions = [], []
        total_loss = 0
        is_train = process == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.no_grad() if not is_train else nullcontext():
            for batch, labels in self.dataloaders[process]:
                loss, predicted = self._process_batch(batch, labels, is_train)
                total_loss += loss
                epoch_labels.append(labels.numpy())
                epoch_predictions.append(predicted.numpy())
        n_obs = len(self.dataloaders[process].dataset)
        accuracy = self._calc_accuracy(epoch_labels, epoch_predictions)
        return [total_loss/n_obs, accuracy], n_obs

    def check_predictions(self, dataset, temperature=1):
        self.model.eval()
        results = []
        with self._device_context(self.model), torch.no_grad():
            for i, (inputs, label) in enumerate(dataset):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs.unsqueeze(0))[0]
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.squeeze().item()
                if label != predicted:
                    if temperature != 1:
                        outputs = outputs / temperature
                    softmax = nn.Softmax(dim=1)
                    results.append((i, softmax(outputs).to("cpu")))
        return results
