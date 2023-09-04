import copy
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Type, Set, List, Dict

import numpy
import torch
from pyhocon import ConfigTree
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.datasets import VisionDataset
import torch.distributed as dist

from utils.general import find_type

DATASET_SPECIFICATIONS = {"base_path": str, "loader": str}


def _worker_init(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.multiprocessing.set_sharing_strategy("file_system")


class IODSplit(VisionDataset, ABC):
    def __init__(self, index, root, transform):
        self.frames: List[Path]
        self.index = index
        self._class_occs = None
        super().__init__(root, transform)

    @property
    @abstractmethod
    def targets(self) -> List[Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def labels(self) -> Set[int]:
        raise NotImplementedError()

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def class_occurrences(self):
        if isinstance(self._class_occs, dict):
            return self._class_occs

        occs = defaultdict(int)
        for t in self.targets:
            for label_idx in t["labels"]:
                occs[label_idx.item()] += 1
        self._class_occs = dict(occs)
        return self._class_occs.copy()

    def get_loader(self, shuffle=False, sampler_drop_last=False, **kwargs):
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=sampler_drop_last,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank())

        loader_kwargs = dict(
            num_workers=0
        )

        loader_kwargs.update(kwargs)
        loader = DataLoader(self, sampler=sampler, worker_init_fn=_worker_init, **loader_kwargs)
        return loader


class IODDataset:
    def __init__(self, cfg: ConfigTree, name=None):
        from data.split_loader import SplitLoader
        self.name = name
        for k, v in DATASET_SPECIFICATIONS.items():
            if k not in cfg:
                raise NameError("Key `{0}` not found in passed configuration for dataset {1}. ".format(k, type(self)) +
                                "Please ensure that all dataset configs contain the required items.")
            if not isinstance(cfg[k], v):
                raise ValueError("{0} passed for key {1}. Please ensure that all {1}s ".format(cfg[k], k) +
                                 "are of type {0}".format(v))
        cls = cfg.get_string("loader")
        loader_cls: Type["SplitLoader"] = find_type(cls)

        self._loader = loader_cls(**cfg)
        self.train_splits, self.test_splits, self.n_splits = self._loader.construct_splits()
        self.class_names = self._loader.class_names

        self.labels = set().union(*self.split_classes)

    @property
    def split_classes(self):
        return self._loader.split_classes

    @property
    def num_classes(self):
        # It is important to consider that the background class is counted in this total.
        return len(self.class_names)

    @property
    def train_split_sizes(self):
        return [len(x) for x in self.train_splits]

    @property
    def eval_split_sizes(self):
        return [len(x) for x in self.test_splits]
