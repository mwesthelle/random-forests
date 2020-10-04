from enum import Enum
from typing import Dict, List, Union

from base_model import BaseModel

import numpy as np


class FeatureSelectionStragegy(Enum):
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
        self.attributes: Dict[str, List[Union[str, float]]] = attributes
        self.outcomes = outcomes
        self.feature_selection_strategy: FeatureSelectionStragegy

    def fit(self, data_iter: List[str], attributes: List[str]):
        attribute2col_map = {k: v for k, v in enumerate(attributes)}
        for row in data_iter:
            *values, class_ = row
            for val_idx, val in enumerate(values):
                attr_name = attribute2col_map[val_idx]
                self.attributes.setdefault(attr_name, []).append(val)
            self.outcomes.append(class_)

    def predict(self, test_data: List[str]):
        pass

    def calculate_info_gain(self):
        pass

    def calculate_feature_info_gain(feature, data_size: int):
        pass

    @staticmethod
    def numerical2categorical(feature):
        feature_mean = np.mean(feature)
        return [
            "greater" if datapoint > feature_mean else "lesser" for datapoint in feature
        ]
