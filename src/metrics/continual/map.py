import operator
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from metrics.storage.boolean import BooleanCache
from src.metrics.base import TMetricResult
from src.metrics.continual.iou import IntersectionOverUnion


class MeanAveragePrecision(IntersectionOverUnion):
    def __init__(self, name: str, exp, result_keys: Dict[str, str], threshold):
        """
        Creates an instance of the standalone IOU metric.

        By default this metric in its initial state will return an IOU
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        super().__init__(name, exp, result_keys)

        """
        The mean utility that will be used to store the running IOU
        for each task label.
        """
        self._caches: Dict[int, BooleanCache] = dict()

        # IoU threshold for TP vs. FP
        self._threshold = threshold

    @torch.no_grad()
    def update(
            self, res: List[Dict]
    ) -> None:
        """
        Update the running accuracy given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :return: None.
        """
        for r in res:
            s_id = self._r_get(r, "split_idx")
            iou = self._get_iou(r)
            # if not isinstance(iou, float) and torch.isnan(iou):
            #     continue
            ev: torch.Tensor = iou >= self._threshold

            labels = self._r_get(r, "gt_labels")
            elems, counts = [x.tolist() for x in labels.unique(return_counts=True)]
            for c, count in zip(elems, counts):
                if c not in self._caches:
                    self._caches[c] = BooleanCache()
                self._caches[c].add_entry(s_id, ev[c], weight=count)

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

    def reset(self):
        super().reset()
        self._caches: Dict[int, BooleanCache] = dict()