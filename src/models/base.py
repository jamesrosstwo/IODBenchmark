from abc import ABC, abstractmethod
from pathlib import Path

from pyhocon import ConfigFactory
from torch import nn

from typing import TYPE_CHECKING, Union, Tuple, Any
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel

from utils.general import find_type

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class IODModel(nn.Module, ABC):
    def __init__(self, exp: "IODExperiment"):
        super().__init__()
        self._device = exp.torch_device
        self._logger = exp.logger

    @abstractmethod
    def forward(self, patterns, targets=None) -> Tuple[Any, Any]:
        """
        :param patterns: A batch of images to feed forward.
        :param targets: Optionally provide targets to receive losses as well (Or purely losses)
        :returns (Predictions, Losses)
        """
        pass

    @classmethod
    def from_checkpoint(_, exp: "IODExperiment", checkpoint_path: Path):
        conf_path = checkpoint_path / "model.conf"
        model_conf = dict(ConfigFactory.parse_file(str(conf_path)))
        cls = find_type(model_conf["cls"])
        del model_conf["cls"]
        out_model: IODModel = cls(exp, **model_conf)
        out_model.restore(exp, checkpoint_path)
        return out_model

    @abstractmethod
    def restore(self, exp: "IODExperiment", checkpoint_path: Path):
        raise NotImplementedError()

    """
    REQUIREMENTS:
    - model.conf with cls pointing to the model class
    - all kwargs required for the model creation
    """

    @abstractmethod
    def checkpoint(self, out_path: Path):
        raise NotImplementedError()

    def img_embeddings(self, patterns):
        raise NotImplementedError("embeddings(patterns) is not implemented for model {0}".format(type(self)))


def img_embeddings(model: Union[IODModel, DistributedDataParallel], *args, **kwargs):
    if isinstance(model, IODModel):
        return model.img_embeddings(*args, **kwargs)
    else:
        module: IODModel = model.module
        assert isinstance(module, IODModel)
        return module.img_embeddings(*args, **kwargs)
