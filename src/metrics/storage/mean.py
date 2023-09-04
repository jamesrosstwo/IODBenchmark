import collections
from typing import Any

from src.metrics.storage.cache import DataCache


class MeanCache(DataCache):
    def add_entry(self, key: collections.Hashable, val: Any, weight: float = 1):
        assert hasattr(val, "__add__")
        assert hasattr(val, "__truediv__")
        super().add_entry(key, (val, weight))

    def __getitem__(self, k):
        entries = self._entries[k]
        res = 0
        div = 0
        for e in entries:
            res += e[0] * e[1]
            div += e[1]
        return res / div
