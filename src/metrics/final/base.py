from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.data.dataset import IODDataset
from src.metrics.base import IODMetric, TMetricResult


class FinalMetric(IODMetric[TMetricResult], ABC):
    def __init__(self, name: str, exp, tracked_metrics: List[str]):
        self._name = name
        self._exp = exp
        self.tracked_metrics = tracked_metrics

    @property
    def name(self):
        return self._name

    @abstractmethod
    def result(self, dataset: IODDataset, label_last_seen_hist: np.array, label_occurrences: np.array,
               **metric_results) -> TMetricResult:
        """
        Evaluate the final metric on all continual metric results for a given dataset
        :param dataset: The IODDataset on which the passed metric results were generated through evaluation
        :param label_last_seen_hist: For each class within dataset, the number of datapoints trained since the class was
        last seen. This information is stored for each split, and should therefore be of size (n_splits, n_classes)
        :param label_occurrences: For each class within the dataset, number of times the class has appeared during
        training. Should be of size (n_classes)
        :param metric_results: Results of the tracked continual metrics in an ndarray
        """
        pass

    def reset(self, **kwargs):
        pass
