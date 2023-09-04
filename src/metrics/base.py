from abc import abstractmethod
from typing import TypeVar, TYPE_CHECKING

from pyhocon import ConfigTree
from typing_extensions import Protocol

# Based on avalanche metrics
from src.utils.general import find_type
if TYPE_CHECKING:
    from experiments.experiment import IODExperiment

TMetricResult = TypeVar("TMetricResult")


class IODMetric(Protocol[TMetricResult]):
    @abstractmethod
    def reset(self, **kwargs):
        pass

    @classmethod
    def from_conf(cls, name: str, conf: ConfigTree, exp: "IODExperiment"):
        if "cls" not in conf:
            raise KeyError("cls not present in metric config. Ensure that a class is specified for all metrics")

        c = conf.get_string("cls")
        conf_d = conf.as_plain_ordered_dict()
        del conf_d["cls"]
        return find_type(c)(name, exp, **conf_d)
