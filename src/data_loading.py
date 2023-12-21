"""A module that automates the loading of the custom dataset used in this
project"""

import glob
import json
import os
from typing import Callable, Optional, Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class StrokeDataPaths:
    """An object that to track raw Japanese canvas data filepaths.
    Args:
        data_directory (str): A directory where the json stroke data is stored.
        class_directory (str): A directory where jsons of the classes are kept.
    """

    def __init__(self, data_directory: str, class_directory: str) -> None:
        self._path_col: str = "path"
        self._label_col: str = "label"
        self._sc_col: str = "stroke_count"
        self._class_char_col: str = "char_id"
        self._class_index_name: str = "index"
        self._class_dicts: dict[str, str] = {"hir": "hiragana.json"}
        self._classes: pd.DataFrame = self._load_class_df(class_directory)
        self._paths: Optional[pd.DataFrame] = None
        self.find_data(data_directory)

    def _load_class_df(self, class_directory: str) -> pd.DataFrame:
        dfs = []
        for slug, path in self._class_dicts.items():
            df = pd.read_json(os.path.join(class_directory, path))
            # Delete this line later when upstream dict is fixed.
            df[self._class_char_col] = df[self._class_char_col].str.replace("-", "", regex=False)
            # Delete this when optional characters are included
            df = df[~df[self._class_char_col].isin(["we", "wi"])].reset_index(drop=True)

            df.insert(0,
                      self._label_col, slug + "_" + df[self._class_char_col])
            #df[self._label_col] = slug + "_" + df[self._class_char_col]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _extract_labels(self, paths: pd.Series) -> pd.DataFrame:
        tmp_col = self._label_col + "tmp"
        idx_col = self._class_index_name
        # Quickly converts the paths to filenames.
        labels = paths.str.rpartition("/").iloc[:, 2]
        # Interrupts if there are any potential problematic json filenames.
        if (labels.str.count("_") != 4).any():
            errors = labels[labels.str.count("_") != 4].values.tolist()
            raise RuntimeError(f"Incorrectly formatted paths at {errors}")
        # Uses the filenames to extract the labels and stroke counts.
        labels = labels.str.extract(r"(^[^_]+_[^_]+)_(\d+)", expand=True)
        labels.columns = [tmp_col, self._sc_col]
        labels.stroke_count = labels.stroke_count.astype(int)
        label_classes = self._classes.reset_index(drop=False)
        label_classes = label_classes.set_index(self._label_col)
        labels[self._label_col] = labels.join(label_classes[idx_col],
                                              on=tmp_col,
                                              how="left")[idx_col].astype(int)
        return labels.drop(columns=[tmp_col])

    def find_data(self, directory: str) -> None:
        """Scans a directory for stroke json data files and saves a dataframe
        of paths, labels, and stroke counts internally.

        Args:
            directory (str): The directory to search.

        Raises:
            ValueError: When there is no directory in the directory path.
            RuntimeError: When the directory does not contain any json data.
        """
        if not os.path.isdir(directory):
            raise ValueError("Directory path is not a valid directory")
        search_string = os.path.join(directory, "**.json")
        paths = pd.Series(glob.glob(search_string), name=self._path_col)
        paths.sort_values(inplace=True)
        paths.reset_index(drop=True, inplace=True)
        if paths.empty:
            raise RuntimeError(f"There are no json files in {directory}")
        self._paths = pd.concat([paths, self._extract_labels(paths)], axis=1)

    def get_classes(self) -> pd.DataFrame:
        """Gets the entire stored Dataframe of the information for each class.

        Returns:
            pd.DataFrame: A pandas Dataframe with columns for the id,
            character, and stroke count of each class.
        """
        return self._classes

    def get_data(self) -> pd.DataFrame:
        """Gets the entire stored Dataframe of dataset paths.

        Returns:
            pd.DataFrame: A pandas Dataframe with columns for the paths,
            labels, and stroke counts of the data.
        """
        return self._paths

    def train_val_test_split(self,
                             test_size: Optional[float] = None,
                             train_size: Optional[float] = None,
                             **kwargs
                             ) -> list[pd.DataFrame]:
        """Wraps sklearn's train test split method in a function using the
        object's stored path DataFrame. Narrows the api by using a single
        array and float sizes. Validation set size is included and inferred
        when both test_size and train_size are specified and have a sum < 1.

        Args:
            test_size (Optional[float], optional): The size of the testing set.
                Defaults to None.
            train_size (Optional[float], optional): The size of the training
                set. Defaults to None.

        Raises:
            RuntimeError: If the paths Dataframe is uninitialized.
            ValueError: If the sum of test_size + train_size > 1.

        Returns:
            list[pd.DataFrame]: The training set and optional testing
            and validations sets.
        """
        if self._paths is None:
            raise RuntimeError("You must run the find_data method using a "
                               "valid directory before performing a split.")
        if train_size is None:
            return train_test_split(self._paths, test_size=test_size, **kwargs)
        if test_size is None or train_size + test_size == 1:
            return train_test_split(self._paths,
                                    train_size=train_size,
                                    **kwargs)
        if test_size + train_size > 1:
            raise ValueError("train size + test_size cannot exceed 1")
        train, test = train_test_split(self._paths,
                                       test_size=test_size,
                                       **kwargs)
        test_size = test_size / (1 - train_size)
        val, test = train_test_split(test, test_size=test_size, **kwargs)
        return [train, val, test]


class StrokeDataset(Dataset):
    """Custom Dataset class that loads jsons from a directory with optional
    methods to restrict based on stroke count and include data from classes
    with higher, but auto-truncated stroke counts.

    Args:
        path_df (pd.DataFrame): A pandas Dataframe containing columns for
            "path", "label", and "stroke_count".
        transform (Optional[Callable], optional): A torch composed
            transform object for prepping the data. Defaults to None.
    """
    def __init__(
            self,
            path_df: pd.DataFrame,
            transform: Optional[Callable] = None
            ) -> None:
        self.paths: npt.NDArray = path_df["path"].values
        self.labels: npt.NDArray[np.int_] = path_df["label"].values
        self.strokes: npt.NDArray[np.int_] = path_df["stroke_count"].values
        self.available_counts = path_df["stroke_count"].unique()
        self.available_counts.sort()
        self.transform = transform
        self._n_strokes: Optional[int] = None
        self._indices: Optional[npt.NDArray[np.int_]] = None

    def __len__(self) -> int:
        if self._n_strokes == -1:
            return 0
        if self._indices is None:
            return self.paths.size
        return self._indices.size

    def _load_json(self,
                   sample_idx: int,
                   ) -> list[list[list[float]]]:
        path = self.paths[sample_idx]
        with open(path, "r", encoding="utf8") as f:
            output = json.load(f)
        return [output[str(i)] for i in range(1, len(output) + 1)]

    def _translate_index(self, idx: int) -> int:
        if self._n_strokes is None:
            return idx
        if self._n_strokes == -1:
            raise IndexError("Unable to index empty dataset selection.")
        # If the datasaet has a stroke count restraint indexes are converted
        # to paths within a smaller subset.
        return self._indices[idx]

    def _stroke_inequality_matches(self,
                                   n: int,
                                   greater: bool
                                   ) -> tuple[int,
                                              Optional[npt.NDArray[np.int_]]]:
        if greater:
            if n >= self.available_counts.max():
                warn("n is set higher than maximum available stroke. "
                     "Dataset will be empty.")
                return -1, None
            return n, np.argwhere(self.strokes > n).squeeze()
        if n <= self.available_counts.min():
            warn("n is set lower than minimum available stroke. "
                 "Dataset will be empty.")
            return -1, None
        return n, np.argwhere(self.strokes < n).squeeze()

    def _stroke_equality_matches(self,
                                 n: int
                                 ) -> tuple[int,
                                            Optional[npt.NDArray[np.int_]]]:
        if n not in self.available_counts:
            warn("n does not match available stroke. "
                 "Dataset will be empty.")
            return -1, None
        return n, np.argwhere(self.strokes == n).squeeze()

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        return self.get_transformed(idx, None)

    def get_path(self, idx: int) -> str:
        """Selects the filepath that leads to the data for the dataset at a
        given index.

        Args:
            idx (int): An index to chose.

        Returns:
            str: A filepath string.
        """
        idx = self._translate_index(idx)
        return self.paths[idx]

    def get_raw(self,
                idx: int,
                max_strokes: Optional[int] = None
                ) -> tuple[list[list[list[float]]], int]:
        """Selects data and labels from the dataset, but bypasses transforms.

        Args:
            idx (int): An index to chose.
            max_strokes (Optional[int]): Trims the amount of loaded strokes to
                this length.

        Returns:
            tuple[list[list[list[float]]], int]: A nested list of x and y
            coordinates for each stroke.
        """
        idx = self._translate_index(idx)
        data = self._load_json(idx)[:max_strokes]
        return data, self.labels[idx]

    def get_transformed(self,
                        idx: int,
                        max_strokes: Optional[int] = None
                        ) -> tuple[list[list[list[float]]], int]:
        """Selects data and labels from the dataset, and includes transforms.

        Args:
            idx (int): An index to chose.
            max_strokes (Optional[int]): Trims the amount of loaded strokes to
                this length.

        Returns:
            tuple[list[list[list[float]]], int]: A nested list of x and y
            coordinates for each stroke.
        """
        idx = self._translate_index(idx)
        data = self._load_json(idx)[:max_strokes]
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx]

    def get_indices_for_strokes(
            self,
            n: Optional[int] = None,
            nonmatch_gt: Optional[bool] = None
            ) -> tuple[Optional[int], Optional[npt.NDArray[np.int_]]]:
        """Finds indices of observations that meet a specific stroke
        constraint.

        Args:
            n (Optional[int], optional): The desired stroke count. If n is set,
                it will act as a filter on the dataset by either eliminating
                obeservations that don't match or exceed the indicated number
                or restricting to only those that are an exact match. Defaults
                to None.
            nonmatch_gt (bool, optional): This specifies the conditioning
                behavior. If set to None, only exact matches of the strokes
                will be shown by the dataset. If set to True, only characters
                with stroke counts higher than n will be shown. If set to
                False, only characters with stroke counts lower than n will be
                shown. Defaults to None.

        Raises:
            ValueError: If n is set to an illegal value <= 0.
        """
        if n is None:
            return None, None
        if n <= 0:
            raise ValueError("n must be set to a value of 1 or greater.")
        if nonmatch_gt is None:
            return self._stroke_equality_matches(n)
        return self._stroke_inequality_matches(n, nonmatch_gt)

    def generate_class_weights(self,
                               scaled: bool = False
                               ) -> npt.NDArray[np.float32]:
        """Generates class weights based on the labels present in the current
        stroke selection.

        Args:
            scaled (bool): Whether or not values should add to one. Defaults
                to False.

        Returns:
            npt.NDArray[np.float32]: An array of weights where 0s will be put
            in place of any missing classes at the time of calculation.
        """
        labels = self.labels[self._indices]
        indices, counts = np.unique(labels, return_counts=True)
        if scaled:
            counts = 1 / counts
            counts = counts / counts.sum()
        else:
            counts = counts.sum() / counts
        # Any label that is not present will be assigned 0.
        weights = np.zeros(self.labels.max() + 1, dtype=np.float32)
        weights[indices] = counts
        return weights

    def set_stroke_count(
            self,
            n: Optional[int] = None,
            nonmatch_gt: Optional[bool] = None
            ) -> None:
        """Restricts the pool of available data based on the desired number of
        strokes.

        Args:
            n (Optional[int], optional): The desired stroke count. If n is set,
                it will act as a filter on the dataset by either eliminating
                obeservations that don't match or exceed the indicated number
                or restricting to only those that are an exact match. Defaults
                to None.
            nonmatch_gt (bool, optional): This specifies the conditioning
                behavior. If set to None, only exact matches of the strokes
                will be shown by the dataset. If set to True, only characters
                with stroke counts higher than n will be shown. If set to
                False, only characters with stroke counts lower than n will be
                shown. Defaults to None.

        Raises:
            ValueError: If n is set to an illegal value <= 0.
        """
        strokes, indices = self.get_indices_for_strokes(n, nonmatch_gt)
        self._n_strokes = strokes
        self._indices = indices
