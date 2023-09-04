import copy
import math
import sys
from typing import List, Dict, Generator

import torch
from pyhocon import ConfigTree
from torchvision.transforms import transforms
from tqdm import tqdm

from loggers.base import MessageType
from models.faster_rcnn import IODFasterRCNN

from src.data.dataset import IODDataset, IODSplit
from src.metrics.base import IODMetric
from src.strategies.base import IODStrategy
from src.utils.general import collate_fn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class NaiveStrategy(IODStrategy):
    is_parallelizable = True

    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        model_args = cfg.get_config("model", default=dict())

        self.model: IODFasterRCNN = IODFasterRCNN(exp, self._num_classes, **model_args)
        # self.model: IODDeformableDETR = IODDeformableDETR(exp, num_classes=self._num_classes)
        # Define the optimizer and the scheduler
        self._optimizer = self.model.construct_optimizer()

        # self._optimizer = torch.optim.AdamW(params, lr=2e-4,
        #                                     weight_decay=1e-4)

        self._batch_size = cfg.get_int("batch_size")
        warmup_factor = 1.0 / 1000
        warmup_iters = min(
            1000, sum([len(s) for s in dataset.train_splits]) // self._batch_size - 1
        )
        self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
        self._scaler = torch.cuda.amp.GradScaler()

        self._transforms = [transforms.ToTensor()]

        self._checkpoint_interval = cfg.get_int("checkpoint_interval", 10)
        self._attempt_restore()

    def _train_instance(self, images, targets):
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            self._logger.log_message(f"Loss is {loss_value}, stopping training", level=MessageType.ERROR)
            self._logger.log_message(loss_dict)
            sys.exit(1)

        self._optimizer.zero_grad()
        self._scaler.scale(losses).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        self._lr_scheduler.step()

        return loss_value


    @torch.no_grad()
    def _eval_batch(self, images, targets, split_idx) -> List[Dict]:
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        detections, _ = self.model(images, targets)

        return [
            dict(
                x=i,
                x_shape=i.shape,
                pred_bbox=d["boxes"],
                pred_labels=d["labels"],
                proposed_class_logits=d["class_logits"],
                # proposed_box_regression=d["box_regression"],
                # box_regression_targets=d["regression_targets"],
                # unfiltered_labels=d["unfiltered_labels"],
                gt_bbox=t["boxes"],
                gt_labels=t["labels"],
                split_idx=split_idx
            )
            for d, t, i in zip(detections, targets, images)
        ]

    def _train_epoch(self, split: IODSplit, epoch: int) -> float:
        if epoch % self._checkpoint_interval == 0:
            self._exp.checkpoint(name="split{0}_epoch{1}".format(split.index, epoch))
        self._logger.log_message("Training split {0}, epoch {1}".format(split.index, epoch))

        self.model.train()
        losses = []

        data_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        for images, targets in tqdm(data_loader):
            losses.append(self._train_instance(images, targets))

        mean_loss = sum(losses) / len(losses)
        return mean_loss

    @torch.no_grad()
    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = split.get_loader(batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        for images, targets in tqdm(match_loader):
            yield self._eval_batch(images, targets, split.index)
