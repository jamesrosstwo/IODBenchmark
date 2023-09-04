import math
import sys
from typing import List, Dict, Generator

import torch
import torchvision
from pyhocon import ConfigTree
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.dataset import IODDataset, IODSplit
from src.loggers.base import IODLogger
from src.metrics.base import IODMetric
from src.strategies.base import IODStrategy
from src.utils.general import collate_fn, reduce_dict
from src.utils.torchvision.roi_heads import roi_forward
from src.utils.torchvision.rpn import rpn_forward


class IncrementalDetectors(IODStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, logger: IODLogger, device: torch.device):
        super().__init__(cfg, dataset, logger, device)
        self._num_classes = len(dataset.labels)
        self._model: FasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
        ).to(self._device)

        # Replace the classifier with a new one, that has "num_classes" outputs
        # 1) Get number of input features for the classifier
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        # 2) Replace the pre-trained head with a new one
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes).to(self._device)

        # The standard torchvision ROI heads don't output the class logits. Swap out this forward function
        # for one that will give us the logits we need for metrics
        def roi_fwd(*args, **kwargs):
            return roi_forward(self._model.roi_heads, *args, **kwargs)

        def rpn_fwd(*args, **kwargs):
            return rpn_forward(self._model.rpn, *args, **kwargs)

        self._model.roi_heads.forward = roi_fwd
        self._model.rpn.forward = rpn_fwd

        # Define the optimizer and the scheduler
        params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005
        )

        self._batch_size = cfg.get_int("batch_size")
        warmup_factor = 1.0 / 1000
        warmup_iters = min(
            1000, sum([len(s) for s in dataset.train_splits]) // self._batch_size - 1
        )
        self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

        self.epochs_per_split = cfg.get_int("epochs_per_split")
        self._transforms = [transforms.ToTensor()]
        self._storage = dict()

    def _train_batch(self, images, targets):
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        loss_dict = self._model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        self._optimizer.zero_grad()
        losses.backward()
        self._optimizer.step()

        self._lr_scheduler.step()

        loss_dict_reduced["loss"] = losses_reduced
        update_d = loss_dict_reduced
        update_d["lr"] = self._optimizer.param_groups[0]["lr"]
        self._storage.update(update_d)

    def _eval_batch(self, images, targets, split_idx) -> List[Dict]:
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        detections = self._model(images, targets)

        return [
            dict(
                x=i,
                x_shape=i.shape,
                pred_bbox=d["boxes"],
                proposed_class_logits=d["class_logits"],
                proposed_box_regression=d["box_regression"],
                box_regression_targets=d["regression_targets"],
                unfiltered_labels=d["unfiltered_labels"],
                gt_bbox=t["boxes"],
                gt_labels=t["labels"],
                split_idx=split_idx
            )
            for d, t, i in zip(detections, targets, images)
        ]

    def train_split(self, split: IODSplit):
        for epoch in range(self.epochs_per_split):
            # TODO: num workers
            data_loader = torch.utils.data.DataLoader(split, batch_size=self._batch_size, shuffle=True,
                                                      collate_fn=collate_fn)

            self._model.train()
            for images, targets in tqdm(data_loader):
                self._train_batch(images, targets)

    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = torch.utils.data.DataLoader(split, batch_size=self._batch_size, shuffle=True,
                                                   collate_fn=collate_fn)
        self._model.eval()

        for images, targets in tqdm(match_loader):
            yield self._eval_batch(images, targets, split.index)
