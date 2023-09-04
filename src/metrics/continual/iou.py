import operator
from typing import Dict, List

import torch

from src.metrics.continual.base import ContinualMetric
from src.metrics.storage.mean import MeanCache


class IntersectionOverUnion(ContinualMetric[float]):
    def __init__(self, name: str, exp, result_keys: Dict[str, str],
                 invert: bool = False):
        """
        Creates an instance of the standalone IOU metric.

        By default this metric in its initial state will return an IOU
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        needed_keys = ["pred_bbox", "gt_bbox", "pred_labels", "gt_labels", "x_shape"]
        super().__init__(name, exp, result_keys, needed_keys)

        self._caches: Dict[int, MeanCache] = dict()
        self.torch_device = self._exp.torch_device

        # 1 - IoU. Useful for final metrics that assume lower = better.
        self.invert = invert

    def _get_iou(self, r: Dict) -> torch.Tensor:
        pred_bboxes = self._r_get(r, "pred_bbox")
        pred_labels = self._r_get(r, "pred_labels")
        true_bboxes = self._r_get(r, "gt_bbox")
        true_labels = self._r_get(r, "gt_labels")

        null_out = torch.ones(81, device=self.torch_device)

        if len(pred_labels) == 0 and len(true_labels) == 0:
            return null_out

        # n = int(torch.max(torch.cat((pred_labels, true_labels))).item() + 1)
        n = self._exp.current_dataset.num_classes

        mask_shape = (n, *self._r_get(r, "x_shape")[1:])
        pred_mask = torch.zeros(mask_shape, dtype=torch.uint8).to(self.torch_device)
        gt_mask = torch.zeros(mask_shape, dtype=torch.uint8).to(self.torch_device)

        # Union of the true and pred bboxes goes into the masks
        for p_b, p_l in zip(pred_bboxes, pred_labels):
            pred_mask[int(p_l.item()), int(p_b[0]):int(p_b[2]), int(p_b[1]):int(p_b[3])] = 1

        for t_b, t_l in zip(true_bboxes, true_labels):
            gt_mask[int(t_l.item()), int(t_b[0]):int(t_b[2]), int(t_b[1]):int(t_b[3])] = 1

        union_mask = torch.clip(pred_mask + gt_mask, min=0, max=1)
        intersection_mask = torch.bitwise_and(pred_mask, gt_mask)

        union_area = torch.sum(union_mask, dim=(1,2))
        intersection_area = torch.sum(intersection_mask, dim=(1,2))

        if torch.sum(union_area) > 0:
            iou = intersection_area / union_area
        else:
            iou = null_out

        if self.invert:
            return 1 - iou
        return iou

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

            labels = self._r_get(r, "gt_labels")
            elems, counts = [x.tolist() for x in labels.unique(return_counts=True)]
            n = sum(counts)
            for c, count in zip(elems, counts):
                if c not in self._caches:
                    self._caches[c] = MeanCache()

                self._caches[c].add_entry(s_id, iou, count / n)

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
        self._map_cache = MeanCache()
