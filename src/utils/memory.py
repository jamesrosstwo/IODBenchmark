from queue import Queue
from typing import Any, Collection, List

import torch


class MemoryBuffer:
    """
    A queue to replay instances during training.
    """
    def __init__(self,  device: torch.device, capacity: int):
        self._capacity = capacity
        self._device = device
        self._memory = Queue(self._capacity)

    def push_single(self, entry: Any):
        if self._memory.full():
            self._memory.get()
        self._memory.put_nowait(entry)

    def push(self, entries: Collection[Any]):
        for e in entries:
            self.push_single(e)

    def get_all(self) -> List:
        return list(self._memory.queue)

    @property
    def capacity(self):
        return self._capacity
