from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Generator, Dict, Tuple, Union

import torch
import torch.distributed as dist
from pyhocon import ConfigTree
from typing import TYPE_CHECKING

from torch.nn.parallel import DistributedDataParallel

from models.base import IODModel

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment
from src.data.dataset import IODDataset, IODSplit
from src.metrics.base import IODMetric


class IODStrategy(ABC):
    # Can this strategy be parallelized across multiple GPUs
    is_parallelizable = True

    def _get_epochs(self, split_idx):
        if isinstance(self._epochs_per_split, int):
            return self._epochs_per_split
        elif isinstance(self._epochs_per_split, list):
            return self._epochs_per_split[split_idx]

    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        self._exp = exp
        self._device = exp.torch_device
        self._cfg = cfg
        self._dataset = dataset
        self._out_path = exp.out_path
        self._num_classes = dataset.num_classes
        self._logger = exp.logger
        self.model: Union[DistributedDataParallel, IODModel] = None
        self._epochs_per_split = self._cfg["epochs_per_split"]
        if not self.is_parallelizable and dist.get_world_size() > 1:
            raise Exception("Tried to use a strategy that does not support multiple GPUs in a multi-gpu setting.")

    @property
    def iod_model(self) -> IODModel:
        if dist.get_world_size() > 1:
            return self.model.module
        return self.model

    def _attempt_restore(self):
        restore_path = self._cfg.get("restore_path", default=None)
        if restore_path is not None:
            restore_path = Path(restore_path)
            assert restore_path.exists() and restore_path.is_dir()
            self.restore(restore_path)

    def train_split(self, split: IODSplit) -> List[float]:
        losses: List[float] = list()
        for epoch in range(self._get_epochs(split.index)):
            epoch_loss = self._train_epoch(split, epoch)
            losses.append(epoch_loss)
            self._logger.log({"split{0}_epoch_loss".format(split.index): epoch_loss})
            self._logger.log_message("Epoch {0} loss: {1}".format(epoch, epoch_loss))
            self._logger.log_message("Epoch {0} loss range: [{1}, {2}]".format(epoch, min(losses), max(losses)))
        return losses

    @abstractmethod
    def _train_epoch(self, split: IODSplit, epoch: int) -> float:
        raise NotImplementedError()

    @torch.no_grad()
    @abstractmethod
    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        pass

    def checkpoint(self, checkpoint_path: Path):
        model_out_path = checkpoint_path / self.model.__class__.__name__
        model_out_path.mkdir(exist_ok=True)
        self.iod_model.checkpoint(model_out_path)

    def restore(self, checkpoint_path: Path):
        self._logger.log_message("Restoring strategy from {0}".format(checkpoint_path))
        self.iod_model.restore(self._exp, checkpoint_path / self.iod_model.__class__.__name__)

    #     self.model = IODModel.from_checkpoint(self._exp, checkpoint_path)

    def before_dataset(self, dataset: IODDataset):
        pass

    def before_train_split(self, split: IODSplit):
        pass

    def after_train_epoch(self, split: IODSplit, epoch: int):
        pass

    def after_train_split(self, split: IODSplit):
        pass

    def before_eval_split(self, split: IODSplit):
        pass

    def after_eval_split(self, split: IODSplit):
        pass

    def after_dataset(self, dataset: IODDataset):
        pass
