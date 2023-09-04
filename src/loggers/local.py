from pathlib import Path
from typing import Dict, TYPE_CHECKING, Tuple

import cv2
import torch
import torch.distributed as dist
from matplotlib.cm import get_cmap

from metrics.continual.base import ContinualMetric
from src.loggers.base import IODLogger, MessageType

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


def get_col(class_idx: int, col_map) -> Tuple[int, int, int]:
    return tuple([x * 255 for x in col_map(class_idx)[:-1]])

class LocalLogger(IODLogger):
    def __init__(self, exp: "IODExperiment"):
        super().__init__(exp)
        self._out_path = exp.out_path
        self._logfile_path = self._out_path / "experiment.log"
        # Distributed setup will need to open this file once for each gpu
        self._logfile_path.touch(exist_ok=dist.get_world_size() > 1)
        self._logfile = open(str(self._logfile_path), "a")

    def log_message(self, msg: str, level: MessageType = MessageType.INFO):
        tag = "[INFO]"
        if tag == MessageType.WARNING:
            tag = "[WARNING]"
        if tag == MessageType.ERROR:
            tag = "[ERROR]"

        if dist.get_world_size() > 1:
            gpu_tag = dist.get_rank()
            tag += " GPU {0}".format(gpu_tag)

        msg = "{0} {1}\n".format(tag, msg)
        print(msg, end="")
        self._logfile.write(msg)
        self._logfile.flush()

    def log(self, info: Dict):
        self.log_message(str(info))

    def log_cont_result(self, metric: ContinualMetric, res: torch.Tensor):
        log_r = {"{0}_split{1}".format(metric.name, idx): v for idx, v in enumerate(res.T)}
        self.log(log_r)

    def draw_bboxes(self, image, boxes: torch.Tensor, labels: torch.Tensor):
        font_face = cv2.FONT_HERSHEY_DUPLEX
        col_map = get_cmap('viridis', self._exp.current_dataset.num_classes)

        for box, label in zip(boxes, labels):
            cls_idx = label
            start_point = int(box[0]), int(box[1])
            end_point = int(box[2]), int(box[3])
            color = get_col(cls_idx, col_map)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            class_name = self._exp.current_dataset.class_names[cls_idx]
            image = cv2.putText(image, class_name, start_point, font_face, 1, color, thickness, cv2.LINE_AA)
        return image

    def log_labels(self, split_id: int, out_prefix: Path, image_loc: Path, boxes, labels, class_names, is_gt=False):

        tag = "gt" if is_gt else "preds"
        self.log_message(
            "Logging labels for split {0} {1}, image: {2}".format(split_id, tag, image_loc))
        self.log_message(
            "\tBoxes: {0}\n"
            "\tLabels: {1}\n"
            "\tClass_names {2}".format(boxes, labels, class_names)
        )

        img = self.draw_bboxes(cv2.imread(str(image_loc)), boxes, labels)
        path = out_prefix / ("{0}_".format(tag) + image_loc.name)
        cv2.imwrite(str(path), img)

    def __del__(self):
        if not self._logfile.closed:
            self._logfile.close()
