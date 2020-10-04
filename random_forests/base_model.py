from abc import ABC, abstractmethod
from typing import Iterator, List


class BaseModel(ABC):
    @abstractmethod
    def fit(self, data_iter: Iterator[str], attributes: List[str]):
        """
        Train model using data_iter
        """

    @abstractmethod
    def predict(self, test_data: List[str]):
        """
        Return predictions given some test_data
        """
