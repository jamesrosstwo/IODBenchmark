from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist

from metrics.continual.base import ContinualMetric
from src.loggers.base import IODLogger, MessageType


class ConsoleLogger(IODLogger):
    def log_message(self, msg: str, level: MessageType = MessageType.INFO):
        tag = "[INFO]"
        if tag == MessageType.WARNING:
            tag = "[WARNING]"
        if tag == MessageType.ERROR:
            tag = "[ERROR]"

        if dist.get_world_size() > 1:
            gpu_tag = dist.get_rank()
            tag += " GPU {0}".format(gpu_tag)

        print(tag, msg)

    def log(self, info: Dict):
        self.log_message(str(info))

    def log_cont_result(self, metric: ContinualMetric, res: torch.Tensor):
        log_r = {"{0}_split{1}".format(metric.name, idx): v for idx, v in enumerate(res.T)}
        self.log(log_r)

    def log_labels(self, split_id: int, out_prefix: Path, image_loc: Path, boxes, labels, class_names, is_gt=False):
        self.log_message(
            "Logging labels for split {0} {1}, image: {2}".format(split_id, "gt" if is_gt else "preds", image_loc))
        self.log_message(
            "\tBoxes: {0}\n"
            "\tLabels: {1}\n"
            "\tClass_names {2}".format(boxes, labels, class_names)
        )
