import random
from collections import defaultdict
from fractions import Fraction
from itertools import chain
from typing import List

from BaseModel import BaseModel

from Metrics import f_measure, precision

import numpy as np


class KFoldCrossValidation:
    """
    Evaluates a model using k-fold cross validation

    Methods
    -------
    _index_dataset(filename: str)
        Build a map from our classes to the indices they appear in, as well as a list
        of offsets for fast access

    generate_stratified_fold(k_folds)
        Given `k_folds` and our map of classes to indices, generate a fold by
        performing a weighted random sample from our indices, with the class
        proportions as weights

    k_fold_cross_validation(
        filename: str,
        k_folds: int,
        repetitions: int,
        nn: int,
        minkowski_p
        ):
        Perform k-fold cross validation with `k_folds` and repeating it `repetitions`
        times on a KNN model with `nn` nearest neighbors using `minkowski_p` order of
        Minkowski's distance generalization, reading the `filename` dataset
    """

    def __init__(self, model: BaseModel):
        self.klass_idxes = defaultdict(
            list
        )  # Holds classes as keys and indices they occur on as values
        self.model = model
        self._line_offsets = []

    def index_dataset(self, filename: str):
        offset: int = 0
        self._line_offsets.clear()
        with open(filename, "rb") as dataset:
            dataset.seek(0)
            offset += len(next(dataset))  # skip header and set offset
            for idx, row in enumerate(dataset):
                self._line_offsets.append(offset)
                offset += len(row)
                values = row.decode("utf-8").strip().split(",")
                self.klass_idxes[values[-1]].append(idx)

    def generate_stratified_fold(self, k_folds: int) -> List[int]:
        """
        Generate a stratified fold by sampling our index map without repetition. The
        fold is represented by a list of indices.
        """
        klass_proportions = {}
        fold_size = len(self._line_offsets) // k_folds
        fold = []
        for klass in self.klass_idxes:
            proportion = Fraction(
                numerator=len(self.klass_idxes[klass]),
                denominator=len(self._line_offsets),
            )
            klass_proportions[klass] = proportion
            random.shuffle(self.klass_idxes[klass])
        for _ in range(fold_size):
            # Choose a random class using the class proportions as weights for the
            # random draw
            chosen_klass = random.choices(
                list(klass_proportions.keys()),
                weights=list(klass_proportions.values()),
                k=1,
            )[0]
            chosen_idx = 0
            try:
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            except IndexError:
                del self.klass_idxes[chosen_klass]
                del klass_proportions[chosen_klass]
                chosen_klass = random.choices(
                    list(klass_proportions.keys()),
                    weights=list(klass_proportions.values()),
                )[0]
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            finally:
                fold.append(chosen_idx)
        return fold

    def kfold_cross_validation(
        self, filename: str, k_folds: int = 10, repetitions: int = 1,
    ):
        results = []
        for i_repetition in range(repetitions):
            with open(filename, "rb") as dataset:
                random.seed(i_repetition * 3)
                self.index_dataset(filename)
                folds = []
                for _ in range(k_folds):
                    fold_rows = []
                    for idx in self.generate_stratified_fold(k_folds):
                        dataset.seek(self._line_offsets[idx])
                        fold_rows.append(dataset.readline().decode("utf-8").strip())
                    folds.append(fold_rows)

                remaining_idxs = []
                for klass in self.klass_idxes:
                    if len(idxes := self.klass_idxes[klass]) > 0:
                        remaining_idxs.extend(idxes)
                self.klass_idxes.clear()
                remaining_data = []
                for idx in remaining_idxs:
                    dataset.seek(self._line_offsets[idx])
                    remaining_data.append(dataset.readline().decode("utf-8").strip())
                folds[-1].extend(remaining_data)

            precisions = []
            f1_scores = []
            fold_idxes = list(range(len(folds)))
            random.shuffle(fold_idxes)
            all_folds_results = []
            for _ in range(k_folds):
                test_fold_idx = fold_idxes.pop()

                test_outcomes = np.array([int(t[-1]) for t in folds[test_fold_idx]])
                train_folds = list(
                    chain(*(folds[:test_fold_idx] + folds[test_fold_idx + 1 :]))
                )
                self.model.load_train_data(train_folds)
                predictions = self.model.predict(folds[test_fold_idx])

                precisions.append(precision(predictions, test_outcomes))
                f1_scores.append(f_measure(predictions, test_outcomes))
                metrics = dict()
                metrics["precision"] = precision(predictions, test_outcomes)
                metrics["f1-score"] = f_measure(predictions, test_outcomes)
                all_folds_results.append(metrics)

            results.append(all_folds_results)
        return results
