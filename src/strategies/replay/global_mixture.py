from abc import abstractmethod

import numpy as np
from sklearn.mixture import GaussianMixture

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
from src.utils.general import collate_fn

from typing import TYPE_CHECKING

from strategies.dists import DistsStrategy, SplitDistribution

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class MixtureDistribution(SplitDistribution):
    @classmethod
    def from_split(cls, model: IODModel, split: IODSplit, n_samples=100, **mixture_kwargs):
        feats_tensors = []
        for batch in SplitDistribution._rand_sample_batches(split, n_samples, batch_size=12):
            feats_tensors.append(model.img_embeddings(batch))

        feats = torch.concat(feats_tensors)
        gmm: GaussianMixture = GaussianMixture(**mixture_kwargs).fit(feats)
        return MixtureDistribution([gmm])

    def __init__(self, mixtures: List[GaussianMixture]):
        self._mixtures: List[GaussianMixture] = mixtures

    def dist(self, other: "MixtureDistribution") -> float:
        raise NotImplementedError()

    def __add__(self, other: "MixtureDistribution"):
        return MixtureDistribution(self._mixtures + other._mixtures)


@abstractmethod
class GlobalMixtureStrategy(DistsStrategy):
    """
    A strategy that estimates a gaussian mixture model of the image features of each split, and uses that to inform
    the distribution shift calculations.
    """
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._dists = {}
        self._dist_mx = np.empty((dataset.n_splits, dataset.n_splits))

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

    def _split_distribution(self, split: IODSplit) -> MixtureDistribution:
        return MixtureDistribution.from_split(self.model, split)

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

    @torch.no_grad()
    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = split.get_loader(batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        for images, targets in tqdm(match_loader):
            yield self._eval_batch(images, targets, split.index)
