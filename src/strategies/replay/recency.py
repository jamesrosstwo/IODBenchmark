from typing import Dict

import torch
from pyhocon import ConfigTree

from data.dataset import IODSplit, IODDataset
from strategies.replay.base import ReplayStrategy


class RecencyReplayStrategy(ReplayStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._store_per_split = cfg.get_int("store_per_split", 10)

    def _update_buffer(self, split: IODSplit):
        indices = torch.randperm(min(len(split), self._store_per_split))
        self._buffer.push([split[i] for i in indices])

    def _pack_memory(self) -> Dict:
        return dict(
            _store_per_split=self._store_per_split,
            _buffer=self._buffer
        )
