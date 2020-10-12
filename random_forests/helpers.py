from itertools import compress, groupby
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def all_equal(iterable: Iterable[T]) -> bool:
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def get_elements_from_data(iterable: Iterable[T], indices: List[int]):
    """
    Given an iterable and a list of indices, return an iterable containing only the
    elements with the given indices
    """
    indices_set = set(indices)
    for row in iterable:
        selectors = [1 if idx in indices_set else 0 for idx in range(len(row))]
        # Need to include the classification target in the selection
        selectors[-1] = 1
        yield [element for element in compress(row, selectors)]
