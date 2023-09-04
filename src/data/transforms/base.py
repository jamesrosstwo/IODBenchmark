from abc import ABC, abstractmethod
from enum import Enum


class IODTransformDomain(Enum):
    """
    Determines the domain of data this transform is applied to.
    If 0, the transformation is applied to all data.
    If 1, the transformation is applied to training data only.
    If 2, the transformation is applied during evaluation steps only.
    """
    ALL = 0
    TRAIN = 1
    EVAL = 2


class IODTransform(ABC):
    def __init__(self, domain: int):
        self.domain: IODTransformDomain = IODTransformDomain(domain)

    @property
    @abstractmethod
    def needs_targets(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
