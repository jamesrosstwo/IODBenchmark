from abc import abstractmethod, ABC

import numpy as np
from torch.utils.data import Subset, DistributedSampler, DataLoader

from experiments.experiment import IODExperiment

import math
import sys
from typing import List, Dict, Generator

import torch
from pyhocon import ConfigTree
from torchvision.transforms import transforms
from tqdm import tqdm

from loggers.base import MessageType
from models.base import IODModel
from models.faster_rcnn import IODFasterRCNN

from src.data.dataset import IODDataset, IODSplit
from src.metrics.base import IODMetric
from src.strategies.base import IODStrategy
from src.utils.general import collate_fn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment

import torch.distributed as dist


class SplitDistribution(ABC):
    @staticmethod
    def _rand_sample_batches(split: IODSplit, n: int, **loader_kwargs) -> Generator:
        indices = torch.randperm(len(split))[:n]
        exemplars = Subset(split, indices)
        sampler = DistributedSampler(exemplars,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank())

        loader = DataLoader(exemplars, sampler=sampler, **loader_kwargs)
        for img, _ in tqdm(loader):
            yield img.detach().cpu()

    @classmethod
    @abstractmethod
    def from_split(cls, model: IODModel, split: IODSplit):
        raise NotImplementedError()

    @abstractmethod
    def dist(self, other: "SplitDistribution") -> float:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other: "SplitDistribution"):
        raise NotImplementedError()


@abstractmethod
class DistsStrategy(IODStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._dists = {}
        self._dist_mx = np.empty((dataset.n_splits, dataset.n_splits))
        self._global_distribution: SplitDistribution = None

        model_args = cfg.get_config("model", default=dict())

        self.model: IODFasterRCNN = IODFasterRCNN(exp, self._num_classes, **model_args)
        # self.model: IODDeformableDETR = IODDeformableDETR(exp, num_classes=self._num_classes)
        # Define the optimizer and the scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.9, nesterov=True
        )

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

        self.epochs_per_split = cfg.get_int("epochs_per_split")
        self._transforms = [transforms.ToTensor()]

        self._checkpoint_interval = cfg.get_int("checkpoint_interval", 10)
        self._attempt_restore()


    @abstractmethod
    def _split_distribution(self, split: IODSplit) -> SplitDistribution:
        raise NotImplementedError()

    def before_train_split(self, split: IODSplit):
        """
        Estimate the distribution using a SplitDistribution
        :param split: IODSplit used for distribution estimation
        """

        split_d = self._split_distribution(split)
        if self._global_distribution is None:
            self._global_distribution = split_d
        else:
            self._global_distribution = self._global_distribution + split_d



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
                gt_bbox=t["boxes"],
                gt_labels=t["labels"],
                split_idx=split_idx
            )
            for d, t, i in zip(detections, targets, images)
        ]

    def train_split(self, split: IODSplit):
        for epoch in range(self.epochs_per_split):
            if epoch % self._checkpoint_interval == 0:
                self._exp.checkpoint(name="split{0}_epoch{1}".format(split.index, epoch))
            data_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
            self._logger.log_message("Training split {0}, epoch {1}".format(split.index, epoch))

            self.model.train()
            losses = []
            for images, targets in tqdm(data_loader):
                losses.append(self._train_instance(images, targets))

            self._logger.log_message("Epoch {0} loss: {1}".format(epoch, sum(losses) / len(losses)))
            self._logger.log_message("Epoch {0} loss range: [{1}, {2}]".format(epoch, min(losses), max(losses)))

    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = split.get_loader(batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        for images, targets in tqdm(match_loader):
            yield self._eval_batch(images, targets, split.index)
