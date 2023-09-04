from pathlib import Path
from typing import Dict, TYPE_CHECKING

import numpy as np
import torch
import wandb

from metrics.continual.base import ContinualMetric
from src.loggers.base import IODLogger, MessageType

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


def _wandb_box_data(boxes, class_ids, class_names, acc=1, loss=0, gt=False):
    key = "ground_truth" if gt else "predictions"
    other_key = "ground_truth" if not gt else "predictions"
    return {
        key: {
            "box_data": [{
                "position": {
                    "minX": int(box[0].item()),
                    "minY": int(box[1].item()),
                    "maxX": int(box[2].item()),
                    "maxY": int(box[3].item())
                },
                "domain": "pixel",
                "class_id": class_id,
                "box_caption": class_name,
                "scores": {
                    "acc": acc,
                    "loss": loss
                }} for box, class_id, class_name in zip(boxes, class_ids, class_names)]
        },
        other_key: {"box_data": []}
    }


class WandBLogger(IODLogger):
    def __init__(self, exp: "IODExperiment", project_name: str):
        super().__init__(exp)
        wandb.init(project=project_name, dir=str(self._logfile_path))

    def log_message(self, msg: str, level: MessageType = MessageType.INFO):
        tag = "[INFO]"
        if tag == MessageType.WARNING:
            tag = "[WARNING]"
        if tag == MessageType.ERROR:
            tag = "[ERROR]"
        print(tag, msg)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def log_cont_result(self, metric: ContinualMetric, res: torch.Tensor):
        dtst = self._exp.current_dataset
        cls_weights = np.zeros((dtst.n_splits, dtst.num_classes))
        for idx, x in enumerate(self._exp.current_dataset.test_splits):
            for cls, count in x.class_occurrences.items():
                cls_weights[idx, cls] = count

        all_classes = [list(x.labels) for x in dtst.test_splits]
        seen_classes = [list(x.labels) for x in dtst.train_splits]

        def agg_res(idx, v, classes):
            return np.nanmean(v[classes] * cls_weights[idx][classes]) / sum(cls_weights[idx])

        split_res = {"{0}_split{1}".format(metric.name, idx): v for idx, v in enumerate(res.T)}
        all_means = {"{0}_split{1}_mean_all".format(metric.name, idx): agg_res(idx, v, all_classes[idx]) for idx, v in
                     enumerate(res.T)}
        seen_means = {"{0}_split{1}_mean_seen".format(metric.name, idx): agg_res(idx, v, seen_classes[idx]) for idx, v
                      in enumerate(res.T)}
        print(split_res)
        self.log(all_means)
        self.log(seen_means)

    def log_labels(self, split_id: int, out_prefix: Path, image_loc: Path, boxes, labels, class_names, is_gt=False):
        box_dict = _wandb_box_data(boxes, labels, class_names, gt=is_gt)
        img = wandb.Image(str(image_loc), boxes=box_dict)
        img_name = "split_{0}_img_{1}_{2}".format(split_id, image_loc.stem, "gt" if is_gt else "pred")
        self.log({img_name: img})

    def __del__(self):
        wandb.finish()
