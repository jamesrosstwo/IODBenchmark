import operator
from typing import List, Dict

import torch

from src.metrics.continual.base import ContinualMetric
from src.metrics.storage.mean import MeanCache


class Loss(ContinualMetric[float]):
    def __init__(self, name: str, exp, result_keys: Dict[str, str]):
        super().__init__(name, exp, result_keys, ["loss"])
        self._caches: Dict[int, MeanCache] = dict()
        self._res_keys = result_keys

    @torch.no_grad()
    def update(
            self, res: List[Dict]
    ) -> None:
        for r in res:
            self._mean_loss.add_entry(r["split_idx"], r[self._res_keys["loss"]])

    def result(self, num_splits, num_classes) -> torch.Tensor:
        out_tensor = torch.empty((num_classes, num_splits), dtype=torch.float64)

        # Splits may not contain data points from every class. We can't evaluate if there is no data,
        # So fill with nan in this case.
        out_tensor[:, :] = float("nan")
        caches = sorted(self._caches.items(), key=operator.itemgetter(0))
        caches = [x[1] for x in caches]

        for i, cache in enumerate(caches):
            for j, val in dict(cache).items():
                out_tensor[i, j] = val

        return out_tensor

    def reset(self, **kwargs) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        self._mean_loss = MeanCache()
