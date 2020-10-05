from collections import Counter
from enum import Enum
from operator import itemgetter
from typing import Dict, List, Tuple, Union, cast

from base_model import BaseModel

import numpy as np


class SelectionStrategy(Enum):
    id3 = 1
    c45 = 2
    cart = 3


class TreeNode:
    def __init__(self):
        self.children = []
        self.is_leaf = None


class DecisionTree(BaseModel):
    def __init__(self, root=None, attributes={}, outcomes=[]):
        self.root = root
        self.attributes: Dict[str, List] = attributes
        self.outcomes: List[Union[str, int]] = outcomes
        self.selection_strategy: SelectionStrategy = SelectionStrategy.c45

    def fit(self, data_iter: List[str], attributes: List[str]):
        attribute2col_map = {k: v for k, v in enumerate(attributes)}
        for row in data_iter:
            *values, class_ = row
            for val_idx, val in enumerate(values):
                attr_name = attribute2col_map[val_idx]
                self.attributes.setdefault(attr_name, []).append(val)
            self.outcomes.append(class_)
        for att in self.attributes:
            self.attributes[att] = DecisionTree.numerical2categorical(
                self.attributes[att]
            )

    def predict(self, test_data: List[str]):
        pass

    def get_best_attribute(self) -> str:
        outcomes_info = DecisionTree.calculate_info(self.outcomes)
        attributes_info = {
            att: self.calculate_attribute_info(att_vals)
            for att, att_vals in self.attributes.items()
        }
        info_gains: List[Tuple[str, float]] = [
            (att, DecisionTree.calculate_info_gain(outcomes_info, att_info))
            for att, att_info in attributes_info.items()
        ]
        return sorted(info_gains, key=itemgetter(1), reverse=True).pop()[0]

    def calculate_attribute_info(self, attribute_data: List[str]) -> float:
        attr_class_counter = Counter(zip(attribute_data, self.outcomes))
        category_counter = Counter(attribute_data)
        classes_ = set(self.outcomes)
        attribute_info = 0
        for category in category_counter:
            category_class_probs = [
                attr_class_counter[(category, class_)] / category_counter[category]
                for class_ in classes_
            ]
            attribute_info += (
                category_counter[category] / len(attribute_data)
            ) * DecisionTree.calculate_entropy(category_class_probs)
        return attribute_info

    @staticmethod
    def calculate_info_gain(outcome_info: float, attribute_info: float) -> float:
        return outcome_info - attribute_info

    @staticmethod
    def calculate_info(data: List[Union[str, int]]) -> float:
        category_counter = Counter(data)
        total_data_points = len(data)
        probs = np.array([v / total_data_points for v in category_counter.values()])
        return DecisionTree.calculate_entropy(probs)

    @staticmethod
    def calculate_entropy(probs):
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def numerical2categorical(attribute_data: List[Union[str, float]]):
        try:
            feature_mean = np.mean(attribute_data)
        except TypeError:
            return cast(List[str], attribute_data)
        return [
            "greater" if datapoint > feature_mean else "lesser"
            for datapoint in attribute_data
        ]
