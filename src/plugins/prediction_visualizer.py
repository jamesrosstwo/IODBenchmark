from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
import torch.distributed as dist

from data.dataset import IODSplit, IODDataset
from experiments.experiment import IODExperiment
from models.base import IODModel
from plugins.base import IODPlugin


def get_col(class_idx: int, col_map) -> Tuple[int, int, int]:
    return tuple([x * 255 for x in col_map(class_idx)[:-1]])


class PredictionVisualizer(IODPlugin):
    def __init__(self, name, exp: IODExperiment, datapoints_per_train_split: int = 5):
        super().__init__(name, exp)
        self._datapoints_per_split = datapoints_per_train_split
        self._cached_data: List = []
        self._out_prefix = self._exp.out_path / "PredictionVisualizer"
        self._out_prefix.mkdir(parents=True, exist_ok=dist.get_world_size() > 1)
        self._col_map: Optional[Colormap] = None
        self._n_classes: int = np.nan
        self._class_names: Dict[int, str] = dict()
        self._split_counts = defaultdict(int)

    def before_dataset(self, dataset: IODDataset):
        self._n_classes = dataset.num_classes
        self._class_names = dataset.class_names
        self._col_map = get_cmap('viridis', self._n_classes)

        for split in dataset.train_splits:
            indices = torch.randperm(len(split))[:self._datapoints_per_split]
            for i in indices:
                self._cached_data.append((split.frames[i], *split[i]))

    def _log_labels(self, split_id: int, out_prefix: Path, image_loc: Path, boxes: torch.Tensor, labels: torch.Tensor,
                    is_gt=False):
        out_boxes = []
        class_indices = []
        class_names = []
        for box, label in zip(boxes, labels):
            cls_idx = label.item()
            name = self._class_names[cls_idx]
            out_boxes.append(box)
            class_indices.append(cls_idx)
            class_names.append(name)

        self._exp.logger.log_labels(split_id, out_prefix, image_loc, out_boxes, class_indices, class_names, is_gt=is_gt)

    def after_eval_split(self, split: IODSplit):
        model: IODModel = self._exp.strategy.model
        model.eval()
        prefix = self._out_prefix / "Split{0}_gpu{1}_{2}".format(split.index, dist.get_rank(),
                                                                 self._split_counts[split.index])
        prefix.mkdir()
        with torch.no_grad():
            for loc, p, t in self._cached_data:
                p = p.to(self._device)
                t = {k: v.to(self._device) for k, v in t.items()}
                preds, _ = model([p], [t])

                p_boxes = preds[0]["boxes"].cpu().detach()
                p_labels = preds[0]["labels"].cpu().detach()

                t_boxes = t["boxes"].cpu().detach()
                t_labels = t["labels"].cpu().detach()
                self._log_labels(split.index, prefix, loc, p_boxes, p_labels)
                self._log_labels(split.index, prefix, loc, t_boxes, t_labels, is_gt=True)

                # pred_path = prefix / ("pred_" + loc.name)
                # gt_path = prefix / ("gt_" + loc.name)
                # cv2.imwrite(str(pred_path), pred_img)
                # cv2.imwrite(str(gt_path), gt_img)
        self._split_counts[split.index] += 1

    def after_train_split(self, split: IODSplit, losses: List[float]):
        self.after_eval_split(split)
