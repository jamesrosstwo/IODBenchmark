from typing import Dict, TYPE_CHECKING

from utils.memory import MemoryBuffer

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment

import torch
from pyhocon import ConfigTree

from data.dataset import IODSplit, IODDataset
from strategies.replay.base import ReplayStrategy
from strategies.replay.buffer import JPEGBuffer


class RandomReplayStrategy(ReplayStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._exemplars_per_split = cfg.get_int("exemplars_per_split", 100)

    def _update_buffer(self, split: IODSplit):
        indices = torch.randperm(min(len(split), self._exemplars_per_split))
        self._buffer.push([split[i] for i in indices])

    def _pack_memory(self) -> Dict:
        return dict(
            _store_per_split=self._exemplars_per_split,
            _buffer=self._buffer.get_all()
        )


class CompressedRandomReplayStrategy(ReplayStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._exemplars_per_split = cfg.get_int("exemplars_per_split", 100)
        self._jpeg_quality = cfg.get_int("jpeg_quality", 50)
        self._buf_size = self._exemplars_per_split * self._dataset.n_splits

    def _initialize_buffer(self) -> MemoryBuffer:
        return JPEGBuffer(self._buf_size, self._device, self._jpeg_quality)

    def _update_buffer(self, split: IODSplit):
        indices = torch.randperm(min(len(split), self._exemplars_per_split))
        self._buffer.push([split[i] for i in indices])

    def _pack_memory(self) -> Dict:
        return dict(
            _store_per_split=self._exemplars_per_split,
            _buffer=self._buffer
        )
