import collections
from abc import ABC, abstractmethod
from typing import Any, List, Dict


class DataCache(ABC):
    def __init__(self):
        self._entries: Dict[collections.Hashable, List[Any]] = dict()

    @abstractmethod
    def __getitem__(self, k):
        pass

    def keys(self):
        return self._entries.keys()

    def add_entry(self, key: collections.Hashable, val: Any):
        if key not in self._entries:
            self._entries[key] = list()
        self._entries[key].append(val)

    def pop_all(self) -> Dict[str, Any]:
        out = dict(self)
        self._entries = dict()
        return out
