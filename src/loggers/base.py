from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch

from metrics.continual.base import ContinualMetric

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class MessageType(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


class IODLogger(ABC):
    def __init__(self, exp: "IODExperiment"):
        self._exp = exp
        self._logfile_path = exp.out_path

    @abstractmethod
    def log_message(self, msg: str, level: MessageType = MessageType.INFO):
        raise NotImplementedError()

    @abstractmethod
    def log(self, info: Dict):
        raise NotImplementedError()

    @abstractmethod
    def log_cont_result(self, metric: ContinualMetric, res: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def log_labels(self, split_id: int, out_prefix: Path, image_loc: Path, boxes, labels, class_names, is_gt=False):
        raise NotImplementedError()
