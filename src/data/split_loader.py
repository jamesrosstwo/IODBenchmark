from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any, Tuple

from data.dataset import IODSplit
from data.transforms.base import IODTransform, IODTransformDomain
from utils.general import find_type


def _construct_transforms(transforms: List[Dict]) -> Tuple[List[IODTransform], List[IODTransform]]:
    train_ts = []
    eval_ts = []
    for t_cfg in transforms:
        cls = find_type(t_cfg["cls"])
        del t_cfg["cls"]
        t: IODTransform = cls(**t_cfg)
        if t.domain in [IODTransformDomain.ALL, IODTransformDomain.TRAIN]:
            train_ts.append(t)
        if t.domain in [IODTransformDomain.ALL, IODTransformDomain.EVAL]:
            eval_ts.append(t)
    return train_ts, eval_ts


class SplitLoader(ABC):
    def _validate_cfg(self, cfg: Dict[str, Any], requirements: Dict[str, Type]):
        for k, v in requirements.items():
            if k not in cfg:
                raise NameError("Key `{0}` not found in passed configuration for dataset {1}. ".format(k, type(self)) +
                                "Please ensure that all dataset configs contain the required items.")
            if not isinstance(cfg[k], v):
                raise ValueError("{0} passed for key {1}. Please ensure that all {1}s ".format(cfg[k], k) +
                                 "are of type {0}".format(v))

    @abstractmethod
    def construct_splits(self) -> List["IODSplit"]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        raise NotImplementedError()
