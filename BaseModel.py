from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    @abstractmethod
    def fit(self, data_iter: List[str]):
        """
        Train model using data_iter
        """

    @abstractmethod
    def predict(self, test_data: List[str]):
        """
        Return predictions given some test_data
        """
