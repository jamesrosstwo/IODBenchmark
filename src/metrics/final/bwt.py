from typing import List

import numpy as np

from src.data.dataset import IODDataset
from src.metrics.final.base import FinalMetric


# For a continual metric we want to maximize, negative BWT signifies forgetting.
# For a continual metric we want to minimize, positive BWT signifies forgetting.
class BackwardsWeightTransfer(FinalMetric[float]):
    def result(self, dataset: IODDataset, label_last_seen_hist: np.array, label_occurrences: np.array, **metric_results):
        T = dataset.n_splits
        bwt = 0
        n = 0

        for metric_idx, res in enumerate(metric_results.values()):
            n += 1
            for i in range(T - 1):
                bwt += res[T - 1][i] - res[i][i]

        return bwt / n
