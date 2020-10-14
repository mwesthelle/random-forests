import math
import random
from collections import Counter
from operator import itemgetter
from typing import Dict, Iterable, List, NewType, Union

from base_model import BaseModel
from decision_tree import DecisionTree, TreeNode
from helpers import get_elements_from_data

DataType = NewType("DataType", Union[str, int, float])
ClassType = NewType("ClassType", Union[str, int])

random.seed(54)


class RandomForest(BaseModel):
    def __init__(self, ntrees: int, model: BaseModel):
        self.ntrees = ntrees
        self.model = model
        self.forest: List[DecisionTree] = []
        self.attr2idx: Dict[str, int]

    def fit(
        self, data_iter: Iterable[List[str]], attribute_names: List[str],
    ):
        """
        Trains the model on data provided by 'data_iter'. The 'attribute_names'
        property MUST be ordered according to the indices in the 'data_iter' lists.
        """
        self.attr2idx = {attr: idx for idx, attr in enumerate(attribute_names)}
        forest_data = [self.generate_bootstrap(data_iter) for _ in range(self.ntrees)]
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
            self.forest.append(DecisionTree(root=TreeNode(attributes_data, outcomes)))
        for tree in self.forest:
            tree.fit()

    def predict(self, test_data: Iterable[List[str]]):
        predictions = []
        votes = []
        for tree in self.forest:
            indices = [self.attr2idx[attr] for attr in tree.root.attribute_data_dict]
            selected_data = get_elements_from_data(test_data, indices)
            votes.append(tree.predict(selected_data))
        for col in zip(*votes):
            prediction = max(Counter(col).items(), key=itemgetter(1))[0]
            predictions.append(prediction)
        return predictions

    @staticmethod
    def generate_bootstrap(data):
        return random.choices(data, k=len(data))
