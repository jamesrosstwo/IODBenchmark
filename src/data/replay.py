from typing import List, Tuple, Set, Dict

import numpy as np
import torch

from data.dataset import IODSplit
from utils.memory import MemoryBuffer


class ReplaySplit(IODSplit):
    @property
    def targets(self) -> List[Dict[str, torch.Tensor]]:
        if self.__targets is None:
            self.__targets = [x[1] for x in self._datapoints]
        return self.__targets

    @property
    def labels(self) -> Set:
        if self.__labels is None:
            self.__labels = set()
            for t in self.targets:
                self.__labels = self.__labels.union(set(t["labels"].tolist()))

        return self.__labels

    def __len__(self) -> int:
        return len(self._datapoints)

    def __init__(self, datapoints: List[Tuple[torch.Tensor, Dict]], root, transform):
        super().__init__(np.nan, root, transform)
        self.__targets = None
        self.__labels = None

        self._datapoints: List[Tuple[torch.Tensor, Dict]] = datapoints

    @classmethod
    def from_buffer(cls, buffer: MemoryBuffer, template_split: IODSplit):
        datapoints: List[Tuple[torch.Tensor, Dict]] = buffer.get_all()
        root = template_split.root
        transform = template_split.transform
        return cls(datapoints, root, transform)

    @classmethod
    def empty(cls, template_split: IODSplit):
        return cls([], template_split.root, template_split.transform)

    def __getitem__(self, idx):
        image, target = self._datapoints[idx]
        return image, target
