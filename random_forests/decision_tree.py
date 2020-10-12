from collections import Counter, OrderedDict
from enum import Enum
from operator import itemgetter
from typing import Dict, Iterable, List, NamedTuple, NewType, Tuple, Union, cast

from base_model import BaseModel
from helpers import all_equal

import numpy as np

DataType = NewType("DataType", Union[str, int, float])
ClassType = NewType("ClassType", Union[str, int])
AttributeVal = NamedTuple("AttributeVal", attribute_name=str, attribute_val=DataType)


class SelectionStrategy(Enum):
    id3 = 1
    c45 = 2
    cart = 3


class TreeNode:
    def __init__(
        self,
        attribute_data_dict: Dict[str, DataType],
        outcomes: List[ClassType],
        best_split_attribute_val: AttributeVal = None,
        idx2attr: Dict[int, str] = dict(),
    ):
        self.best_split_attribute_val = best_split_attribute_val
        self.attribute_data_dict = attribute_data_dict
        self.outcomes = outcomes
        self.children: List[TreeNode] = []
        self.class_ = None
        self.idx2attr = idx2attr


class DecisionTree(BaseModel):
    def __init__(self, root: TreeNode = None):
        self.root = root
        self.selection_strategy: SelectionStrategy = SelectionStrategy.c45

    # TODO: investigate how to handle different columns of different types (categorical
    # or numerical) and remove the ugly hack of the 'numerical' argument
    def fit(
        self,
        data_iter: Iterable[List[str]] = None,
        attribute_names: List[str] = None,
        numerical=False,
    ):
        if self.root is None and data_iter is not None and attribute_names is not None:
            idx2attr = {idx: name for idx, name in enumerate(attribute_names)}
            attributes_data: Dict[str, List[DataType]] = dict()
            outcomes: List[ClassType] = []
            for row in data_iter:
                *values, class_ = row
                outcomes.append(ClassType(class_))
                for val_idx, val in enumerate(values):
                    attr_name = idx2attr[val_idx]
                    attributes_data.setdefault(attr_name, []).append(DataType(val))
            # TODO: this is dumb. remove this and do it proper
            if numerical:
                for att in attributes_data:
                    attributes_data[att] = DecisionTree.numerical2categorical(
                        attributes_data[att]
                    )
            root_node = TreeNode(attributes_data, outcomes, idx2attr=idx2attr)
            self.root = root_node
            self.build_tree(root_node)
        elif self.root is not None:
            self.root.idx2attr = {
                idx: name
                for idx, name in enumerate(self.root.attribute_data_dict.keys())
            }
            self.build_tree(self.root)
        else:
            raise ValueError(
                "Need either a tree node, or all required data to build one"
            )

    def build_tree(self, node: TreeNode):
        if all_equal(node.outcomes):
            node.class_ = node.outcomes[0]
            return
        elif len(node.attribute_data_dict) == 0:
            outcomes_counter = Counter(node.outcomes)
            node.class_ = max(outcomes_counter.items(), key=itemgetter(1))[0]
            return
        else:
            chosen_attribute = self.get_best_attribute(
                node.attribute_data_dict, node.outcomes
            )
            children = self.split_attribute(node, chosen_attribute)
            node.children = children
            for child in children:
                self.build_tree(child)

    def predict(self, test_data: Iterable[List[str]]):
        predictions: List[ClassType] = []
        for row in test_data:
            test_data_point = {
                self.root.idx2attr[col_idx]: col_val
                for col_idx, col_val in enumerate(row[:-1])
            }
            predictions.append(
                self._predict(cast(TreeNode, self.root), test_data_point)
            )
        return predictions

    def get_best_attribute(
        self, attribute_data_dict: Dict[str, DataType], outcomes: List[ClassType]
    ) -> str:
        outcomes_info = DecisionTree.calculate_info(outcomes)
        attribute2info_data = {
            att: (self.calculate_attribute_info(att_vals, outcomes), att_vals)
            for att, att_vals in attribute_data_dict.items()
        }
        gain_ratios: List[Tuple[str, float]] = [
            (att, DecisionTree.calculate_gain_ratio(outcomes_info, att_info, att_data))
            for att, (att_info, att_data) in attribute2info_data.items()
        ]
        return max(gain_ratios, key=itemgetter(1))[0]

    @staticmethod
    def _predict(tree_node: TreeNode, data_point: Dict[ClassType, DataType]):
        if tree_node.class_ is not None:
            return tree_node.class_
        else:
            best_attribute = tree_node.children[0].best_split_attribute_val
            for child in tree_node.children:
                if (
                    data_point[best_attribute.attribute_name]
                    == child.best_split_attribute_val.attribute_val
                ):
                    data_point.pop(child.best_split_attribute_val.attribute_name)
                    return DecisionTree._predict(child, data_point)

    @staticmethod
    def split_attribute(node: TreeNode, chosen_attribute: str):
        ordered_data = OrderedDict(node.attribute_data_dict)
        attribute2idx_map = {
            attr: idx for idx, attr in enumerate(list(ordered_data.keys()))
        }
        idx2attribute_map = {
            idx: attr for idx, attr in enumerate(list(ordered_data.keys()))
        }
        attribute_values = [v for _, v in ordered_data.items()]
        chosen_attribute_set = set(ordered_data[chosen_attribute])
        children_data_dict: Dict[str, Dict[str, List[DataType]]] = dict()
        class_outcomes: Dict[str, List[ClassType]] = dict()
        children: List[TreeNode] = []
        for row_idx, row_vals in enumerate(zip(*attribute_values)):
            chosen_attribute_val = row_vals[attribute2idx_map[chosen_attribute]]
            children_data_dict.setdefault(chosen_attribute_val, dict())
            class_outcomes.setdefault(chosen_attribute_val, []).append(
                node.outcomes[row_idx]
            )
            for col_idx, col_val in enumerate(row_vals):
                if col_idx != attribute2idx_map[chosen_attribute]:
                    children_data_dict[chosen_attribute_val].setdefault(
                        idx2attribute_map[col_idx], []
                    ).append(col_val)
        for attr_value in chosen_attribute_set:
            child_node = TreeNode(
                attribute_data_dict=children_data_dict[attr_value],
                outcomes=class_outcomes[attr_value],
                best_split_attribute_val=AttributeVal(chosen_attribute, attr_value),
            )
            children.append(child_node)
        return children

    @staticmethod
    def calculate_entropy(probs):
        return -np.nan_to_num(np.sum(probs * np.log2(probs)))

    @staticmethod
    def calculate_info(data: List[Union[str, int]]) -> float:
        category_counter = Counter(data)
        total_data_points = len(data)
        probs = np.array([v / total_data_points for v in category_counter.values()])
        return DecisionTree.calculate_entropy(probs)

    # TODO: handle numerical attributes here
    @staticmethod
    def calculate_attribute_info(
        attribute_data: List[str], outcomes: List[str]
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
    def calculate_gain_ratio(
        attribute_info: float, outcome_info: float, attribute_data: List[DataType]
    ) -> float:
        info_gain = DecisionTree.calculate_info_gain(outcome_info, attribute_info)
        category_counter = Counter(attribute_data)
        probs = np.array([v / len(attribute_data) for v in category_counter.values()])
        split_info = DecisionTree.calculate_entropy(probs)
        return info_gain / split_info

    @staticmethod
    def numerical2categorical(attribute_data: List):
        feature_mean = np.mean(attribute_data)
        return [
            "greater" if datapoint > feature_mean else "lesser"
            for datapoint in attribute_data
        ]
