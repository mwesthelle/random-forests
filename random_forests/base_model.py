from abc import ABC, abstractmethod
from typing import Iterable, List


class BaseModel(ABC):
    @abstractmethod
    def fit(self, data_iter: Iterable[List[str]], attributes: List[str]):
        """
        Train model using data_iter
        """

    @abstractmethod
    def predict(self, test_data: List[List[str]]):
        """
        Return predictions given some test_data
        """
