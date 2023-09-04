from typing import List, Any, Dict, TYPE_CHECKING
from abc import abstractmethod
from dataclasses import dataclass
import torch

from src.metrics.base import IODMetric, TMetricResult

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class ContinualMetric(IODMetric[TMetricResult]):
    def __init__(self, name, exp, result_keys: Dict[str, str], needed_keys: List[str]):
        self._name = name
        self._exp: IODExperiment = exp

        """
        Keys determining which part of the batch result to pull from
        """
        self._res_keys = result_keys
        for k in needed_keys:
            if k not in self._res_keys:
                raise KeyError("{0} not found in the passed config for {1}.".format(k, type(self)))

    @property
    def name(self):
        return self._name

    def _r_get(self, res, key):
        try:
            k = self._res_keys[key]
        except KeyError as e:
            print("[ERROR] key {0} was not found in the metric keys. ".format(key) +
                  " Please check your metric configuration.")
            raise e
        try:
            return res[k]
        except KeyError as e:
            print("[ERROR] key {0} was not found in the result passed from the".format(k) +
                  " selected strategy. Check your configuration and ensure they match up.")
            raise e

    @abstractmethod
    def update(self, data: List[Dict]):
        raise NotImplementedError()

    @abstractmethod
    def result(self, num_splits, num_classes) -> torch.Tensor:
        """
        Returns a result tensor containing the evaluation results on all splits, and for each split, results of all classes.
        The result of this metric should therefore be a 2D tensor, the number of splits by the number of classes.
        """
        raise NotImplementedError()


@dataclass
class ContinualMetricResultMatrix:
    """
    Stores the results of a metric over a dataset in a matrix.

    Axis 0: represents the training split index. Index zero means that the strategy has been trained on split zero only.
    Split one would be once the strategy has trained on both split zero and one.

    Axis 1: Represents the evaluation split index.

    Therefore metricResult[2][0] would be the result of evaluating the metric on split 0, after training on splits
    0, 1, and 2.
    """
    result_mx: List[List[Any]]
