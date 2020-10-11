from collections import Counter, OrderedDict
from enum import Enum
from operator import itemgetter
from typing import Dict, List, NewType, Tuple, Union

from base_model import BaseModel
from helpers import all_equal

import numpy as np

DataType = NewType("DataType", Union[str, int, float])
ClassType = NewType("ClassType", Union[str, int])


class SelectionStrategy(Enum):
    id3 = 1
    c45 = 2
    cart = 3


class TreeNode:
    def __init__(
        self,
        attribute_data: Dict[str, DataType],
        outcomes: List[ClassType],
        best_split_attribute: str = None,
    ):
        self.best_split_attribute = best_split_attribute
        self.attribute_data = attribute_data
        self.outcomes = outcomes
        self.children = []
        self.class_ = None


class DecisionTree(BaseModel):
    def __init__(self, root: TreeNode = None):
        self.root = root
        self.selection_strategy: SelectionStrategy = SelectionStrategy.c45

    # TODO: investigate how to handle different columns of different types (categorical
    # or numerical) and remove the ugly hack of the 'numerical' argument
    def fit(self, data_iter: List[str], attribute_names: List[str], numerical=False):
        attribute2col_map = {k: v for k, v in enumerate(attribute_names)}
        attributes_data: Dict[str, List[DataType]] = dict()
        outcomes: List[ClassType] = []
        for row in data_iter:
            *values, class_ = row
            outcomes.append(ClassType(class_))
            for val_idx, val in enumerate(values):
                attr_name = attribute2col_map[val_idx]
                attributes_data.setdefault(attr_name, []).append(DataType(val))
        if numerical:
            for att in attributes_data:
                attributes_data[att] = DecisionTree.numerical2categorical(
                    attributes_data[att]
                )
        root_node = TreeNode(attributes_data, outcomes)
        self.root = root_node
        self.build_tree(root_node)

    def build_tree(self, node: TreeNode):
        if all_equal(node.outcomes):
            node.class_ = node.outcomes[0]
            return
        else:
            attribute_samples_number = int(np.sqrt(len(node.attribute_data)))
            sampled_attributes = np.random.choice(
                list(node.attribute_data.keys()), size=attribute_samples_number
            )
            sampled_attribute_data = {
                k: node.attribute_data[k] for k in sampled_attributes
            }
            chosen_attribute = self.get_best_attribute(
                sampled_attribute_data, node.outcomes
            )
            children = self.split_attribute(node, chosen_attribute)
            node.children = children
            for child in children:
                self.build_tree(child)

    @staticmethod
    def split_attribute(node: TreeNode, chosen_attribute: str):
        ordered_data = OrderedDict(node.attribute_data)
        attribute2idx_map = {
            attr: idx for idx, attr in enumerate(list(ordered_data.keys()))
        }
        idx2attribute_map = {
            idx: attr for idx, attr in enumerate(list(ordered_data.keys()))
        }
        attribute_values = [v for _, v in ordered_data.items()]
        chosen_attribute_set = set(ordered_data[chosen_attribute])
        children_data: Dict[str, Dict[str, List[DataType]]] = dict()
        class_outcomes: Dict[str, List[ClassType]] = dict()
        children: List[TreeNode] = []
        for row_idx, row_vals in enumerate(zip(*attribute_values)):
            chosen_attribute_val = row_vals[attribute2idx_map[chosen_attribute]]
            children_data.setdefault(chosen_attribute_val, dict())
            class_outcomes.setdefault(chosen_attribute_val, []).append(
                node.outcomes[row_idx]
            )
            for col_idx, col_val in enumerate(row_vals):
                if col_idx != attribute2idx_map[chosen_attribute]:
                    children_data[chosen_attribute_val].setdefault(
                        idx2attribute_map[col_idx], []
                    ).append(col_val)
        for attr_value in chosen_attribute_set:
            child_node = TreeNode(
                attribute_data=children_data[attr_value],
                outcomes=class_outcomes[attr_value],
                best_split_attribute=attr_value,
            )
            children.append(child_node)
        return children

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
    def numerical2categorical(attribute_data: List):
        feature_mean = np.mean(attribute_data)
        return [
            "greater" if datapoint > feature_mean else "lesser"
            for datapoint in attribute_data
        ]
