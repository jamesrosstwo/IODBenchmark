import collections

import torch

from src.metrics.storage.cache import DataCache


class TensorStackCache(DataCache):
    def __getitem__(self, k):
        self._stack_fn(self._entries[k])

    def __init__(self, cuda=True):
        self._stack_fn = lambda v: torch.stack(v)
        if cuda:
            self._stack_fn = lambda v: torch.stack(v).cuda()
        super().__init__()

    def add_entry(self, key: collections.Hashable, val: torch.Tensor):
        assert isinstance(val, torch.Tensor)
        super().add_entry(key, val)
