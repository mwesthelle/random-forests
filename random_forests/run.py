import os

from decision_tree import DecisionTree
from random_forest import RandomForest
from kfold_crossvalidation import KFoldCrossValidation

if __name__ == "__main__":
    model = RandomForest(10, DecisionTree())
    # model = DecisionTree()
    kfold = KFoldCrossValidation(model, delimiter=";")
    kfold.kfold_cross_validation(os.path.join(os.getcwd(), "../data", "benchmark.csv"))
