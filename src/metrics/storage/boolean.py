import collections
from src.metrics.storage.cache import DataCache


class BooleanCache(DataCache):
    """
    Finds #true / #total
    """

    def add_entry(self, key: collections.Hashable, val: bool, weight: float = 1):
        super().add_entry(key, (val, weight))

    def __getitem__(self, k):
        entries = self._entries[k]
        res = 0
        div = 0
        for e in entries:
            res += e[0] * e[1]
            div += e[1]
        return res / div
