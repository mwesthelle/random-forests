import math
import random
from typing import Dict, Iterable, List, NewType, Union

from base_model import BaseModel
from decision_tree import DecisionTree, TreeNode

import numpy as np

DataType = NewType("DataType", Union[str, int, float])
ClassType = NewType("ClassType", Union[str, int])

random.seed(54)


class RandomForest(BaseModel):
    def __init__(self, ntrees: int, model: BaseModel):
        self.ntrees = ntrees
        self.model = model

    def fit(
        self,
        data_iter: Iterable[List[str]],
        attribute_names: List[str],
        numerical=False,
    ):
        forest_data = [self.generate_bootstrap(data_iter) for _ in range(self.ntrees)]
        forest: List[DecisionTree] = []
        for tree_data in forest_data:
            idx2attr = {idx: name for idx, name in enumerate(attribute_names)}
            attributes_data: Dict[str, List[DataType]] = dict()
            outcomes: List[ClassType] = []
            attributes_sample_size = int(math.sqrt(len(attribute_names)))
            sample_attribute_indices = set(
                random.choices(list(idx2attr.keys()), k=attributes_sample_size)
            )
            for row in tree_data:
                *values, target = row
                outcomes.append(ClassType(target))
                for val_idx, val in enumerate(values):
                    if val_idx in sample_attribute_indices:
                        attr_name = idx2attr[val_idx]
                        attributes_data.setdefault(attr_name, []).append(DataType(val))
            if numerical:
                for att in attributes_data:
                    attributes_data[att] = DecisionTree.numerical2categorical(
                        attributes_data[att]
                    )
            forest.append(DecisionTree(root=TreeNode(attributes_data, outcomes)))
        for tree in forest:
            tree.fit()

    def predict(self, test_data: Iterable[List[str]]):
        pass

    @staticmethod
    def generate_bootstrap(data):
        return random.choices(data, k=len(data))
