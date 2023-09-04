from pathlib import Path
from typing import List, Generator, Dict

import numpy as np
import torch
from munch import DefaultMunch
from pyhocon import ConfigTree
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data.dataset import IODSplit, IODDataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment
from metrics.base import IODMetric
from strategies.base import IODStrategy
from strategies.context_transformer.data import BaseTransform, preproc
from strategies.context_transformer.layers import PriorBox, Detect
from strategies.context_transformer.layers.modules.multibox_loss_combined import MultiBoxLoss_combined
from strategies.context_transformer.models.RFB_Net_vgg import build_net
from strategies.context_transformer.utils.checkpointer import DetectionCheckpointer, PeriodicCheckpointer
from strategies.context_transformer.utils.nms_wrapper import nms
from utils.general import collate_fn
from strategies.context_transformer.utils.solver import build_optimizer, build_lr_scheduler


class ContextTransformer(IODStrategy):
    def checkpoint(self):
        pass

    is_parallelizable = True

    # TODO: adapt the init_reweight
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)

        self._batch_size = cfg.get_int("batch_size")
        self.epochs_per_split = cfg.get_int("epochs_per_split")
        self._latest_split_id = 0
        self._storage = dict()
        self._dataset = dataset

        self._args = dict(cfg)
        self._eval_args = self._args["eval"]

        priorbox = PriorBox(self._args["dtst"])
        with torch.no_grad():
            self._priors = priorbox.forward().to(self._device)

        arg_bunch = DefaultMunch.fromDict(self._args)

        self._model = build_net(arg_bunch, arg_bunch.dtst.min_dim, dataset.num_classes).to(device)
        if self._device.type == "cuda" and arg_bunch.ngpu > 1:
            self._model = DistributedDataParallel(self._model)
        self._optimizer = build_optimizer(arg_bunch, self._model)
        self._scheduler = build_lr_scheduler(arg_bunch, self._optimizer)

        self._checkpointer = DetectionCheckpointer(
            self._model, arg_bunch, optimizer=self._optimizer, scheduler=self._scheduler
        )

        self._criterion = MultiBoxLoss_combined(dataset.num_classes + 1, 0.5, True, 0, True, 3, 0.5, False)

        start_iter = (
                self._checkpointer.resume_or_load(
                    arg_bunch.basenet if arg_bunch.phase == 1 else arg_bunch.load_file,
                    resume=False).get("iteration", -1) + 1
        )
        self._current_iter = start_iter

        max_iter = arg_bunch.max_iter
        self._periodic_checkpointer = PeriodicCheckpointer(
            self._checkpointer, arg_bunch.checkpoint_period, max_iter=max_iter
        )

        rgb_means = (104, 117, 123)
        self._img_preproc = preproc(arg_bunch.dtst.min_dim, rgb_means, arg_bunch.p)
        self._transform = BaseTransform(self._model.size, rgb_means, (2, 0, 1))
        self._detector = Detect(self._dataset.num_classes + 1, 0, cfg["dtst"])

    def _train_batch(self, images, targets):
        # TODO: Not integrated: EventStorage, max iters
        self._current_iter += 1
        x = torch.stack(images, 0).to(self._device)
        output = self._model(x)
        loss_dict = self._criterion(output, self._priors, targets)
        losses = sum(loss for loss in loss_dict.values())
        self._optimizer.zero_grad()
        losses.backward()
        self._optimizer.step()

        if self._args["phase"] == 2 and self._args["method"] == 'ours':
            if isinstance(self._model, (DistributedDataParallel, DataParallel)):
                self._model.module.normalize()
            else:
                self._model.normalize()
        self._scheduler.step()
        self._periodic_checkpointer.step(self._current_iter)

    def _eval_batch(self, all_boxes, i, images, targets, split_idx) -> List[Dict]:

        ret = []

        # TODO: Replace for loop with tensor operations to allow for parallelization.
        for img, target in zip(images, targets):
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]]).to(self._device)
            with torch.no_grad():
                x = self._transform(img).unsqueeze(0).to(self._device)

            pred = self._model(x)  # forward pass
            boxes, scores = self._detector.forward(pred, self._priors)
            boxes = boxes[0]  # percent and point form detection boxes
            scores = scores[0]  # [1, num_priors, num_classes]

            boxes *= scale  # scale each detection back up to the image
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()

            for j in range(1, self._dataset.num_classes):
                inds = np.where(scores[:, j] > self._eval_args["thresh"])[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

                keep = nms(c_dets, 0.45, force_cpu=self._device.type == "cpu")
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets

            all_pred_boxes = []

            max_per_image = self._eval_args["max_per_image"]
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, self._dataset.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, self._dataset.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
                        all_pred_boxes.append(all_boxes[j][i])

            if len(all_pred_boxes) == 0:
                all_pred_boxes = np.empty((0, 4))
            else:
                all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)



            ret.append(dict(
                x_shape=img.shape,
                pred_bbox=torch.from_numpy(all_pred_boxes).to(self._device),
                pred_labels=torch.ones((all_pred_boxes.shape[0]), device=self._device, dtype=torch.float64),
                proposed_class_logits=torch.from_numpy(scores[:, 1:]).to(self._device),
                proposed_box_regression=torch.from_numpy(boxes).to(self._device),
                # box_regression_targets=d["regression_targets"],
                # unfiltered_labels=d["unfiltered_labels"],
                gt_bbox=target[:, :-2].to(self._device),
                gt_labels=target[:, -2].to(self._device),
                split_idx=split_idx
            ))

        return ret

    def train_split(self, split: IODSplit):
        data_loader = torch.utils.data.DataLoader(split, batch_size=self._batch_size, shuffle=True,
                                                  collate_fn=collate_fn, drop_last=True)
        self._model.train()
        for images, targets in tqdm(data_loader):
            self._train_batch(images, targets)

    @torch.no_grad()
    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = torch.utils.data.DataLoader(split, shuffle=True,
                                                   collate_fn=collate_fn, drop_last=True)
        self._model.eval()

        num_images = len(split)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self._dataset.num_classes)]

        current_i = 0
        for images, targets in tqdm(match_loader):
            eval_res = self._eval_batch(all_boxes, current_i, images, targets, split.index)
            current_i += 1
            yield eval_res
