from collections import Counter
from enum import Enum
from operator import itemgetter
from typing import Dict, List, NewType, Tuple, Union

from base_model import BaseModel
from helpers import all_equal

import numpy as np

DataType = NewType("DataType", Union[str, int, float])
ClassType = NewType("DataType", Union[str, int])


class SelectionStrategy(Enum):
    id3 = 1
    c45 = 2
    cart = 3


class TreeNode:
    def __init__(self, is_leaf):
        self.children = []
        self.is_leaf = is_leaf


class DecisionTree(BaseModel):
    def __init__(self, root: TreeNode = None):
        self.root = root
        self.selection_strategy: SelectionStrategy = SelectionStrategy.c45

    # TODO: investigate how to handle different columns of different types (categorical
    # or numerical) and remove the ugly hack of the 'numerical' argument
    def fit(self, data_iter: List[str], attribute_names: List[str], numerical=True):
        attribute2col_map = {k: v for k, v in enumerate(attribute_names)}
        attributes_data: Dict[str, List[DataType]] = dict()
        outcomes: List[ClassType] = []
        for row in data_iter:
            *values, class_ = row
            outcomes.append(class_)
            for val_idx, val in enumerate(values):
                attr_name = attribute2col_map[val_idx]
                attributes_data.setdefault(attr_name, []).append(val)
        if numerical:
            for att in attributes_data:
                attributes_data[att] = DecisionTree.numerical2categorical(
                    attributes_data[att]
                )

    def build_tree(self, attributes, outcomes, selection_stragegy: SelectionStrategy):
        pass

    def predict(self, test_data: List[str]):
        pass

    def get_best_attribute(self, attribute_data, outcomes) -> str:
        outcomes_info = DecisionTree.calculate_info(outcomes)
        attributes_info = {
            att: self.calculate_attribute_info(att_vals, outcomes)
            for att, att_vals in attribute_data.items()
        }
        info_gains: List[Tuple[str, float]] = [
            (att, DecisionTree.calculate_info_gain(outcomes_info, att_info))
            for att, att_info in attributes_info.items()
        ]
        return sorted(info_gains, key=itemgetter(1), reverse=True).pop()[0]

    def calculate_attribute_info(
        self, attribute_data: List[str], outcomes: List[str]
    ) -> float:
        attr_class_counter = Counter(zip(attribute_data, outcomes))
        category_counter = Counter(attribute_data)
        classes_ = set(outcomes)
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
        feature_mean = np.mean(attribute_data)
        return [
            "greater" if datapoint > feature_mean else "lesser"
            for datapoint in attribute_data
        ]
