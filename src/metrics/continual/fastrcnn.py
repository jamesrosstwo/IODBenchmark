import operator
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.metrics.continual.base import ContinualMetric
from src.metrics.storage.mean import MeanCache


class FastRCNNLoss(ContinualMetric[float]):
    def __init__(self, name: str, exp, result_keys: Dict[str, str], class_weight: float = 1.0, box_weight: float = 1.0):
        needed_keys = ["box_regression", "regression_targets", "class_logits", "labels"]
        super().__init__(name, exp, result_keys, needed_keys)
        self._caches: Dict[int, MeanCache] = dict()
        self._class_weight = class_weight
        self._box_weight = box_weight

    @torch.no_grad()
    def update(
            self, result: List[Dict]
    ) -> None:
        box_regression = torch.cat([self._r_get(r, "box_regression") for r in result])
        regression_targets = torch.cat([self._r_get(r, "regression_targets") for r in result])
        class_logits = torch.cat([self._r_get(r, "class_logits") for r in result])
        labels = torch.cat([self._r_get(r, "labels") for r in result])
        s_id = result[0]["split_idx"]
        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()
        final_loss = box_loss * self._box_weight + classification_loss * self._class_weight

        elems, counts = [x.tolist() for x in labels.unique(return_counts=True)]
        n = sum(counts)
        for c, count in zip(elems, counts):
            if c not in self._caches:
                self._caches[c] = MeanCache()

            self._caches[c].add_entry(s_id, final_loss, count / n)

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
        self._caches = dict()
