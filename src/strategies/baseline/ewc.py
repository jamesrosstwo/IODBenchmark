import math
import sys
from collections import defaultdict
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
from src.utils.avalanche.training import copy_params_dict, zerolike_params_dict
from src.utils.general import collate_fn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class ElasticWeightConsolidation(IODStrategy):
    is_parallelizable = True

    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._num_classes = dataset.num_classes

        model_args = cfg.get_config("model", default=dict())
        self.model: IODFasterRCNN = IODFasterRCNN(exp, self._num_classes, **model_args)

        # Define the optimizer and the scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
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

        self._epochs_per_split = cfg["epochs_per_split"]
        self._transforms = [transforms.ToTensor()]

        # EWC
        self.ewc_lambda = cfg.get_float("lambda")
        self.decay_factor = cfg.get_float("decay_factor")
        self.is_separate = cfg.get_bool("separate_penalties")
        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)
        self._latest_split_id = 0
        self._current_train_split = 0
        self._checkpoint_interval = cfg.get_int("checkpoint_interval", 10)
        self._attempt_restore()

    def _get_epochs(self, split_idx):
        if isinstance(self._epochs_per_split, int):
            return self._epochs_per_split
        elif isinstance(self._epochs_per_split, list):
            return self._epochs_per_split[split_idx]

    def get_penalty(self, split_index: int):
        # Adapted from avalanche
        penalty = torch.tensor(0, device=self._device).float()

        if self.is_separate:
            for experience in range(split_index):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        self.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = saved_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            prev_exp = split_index - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    self.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = saved_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        return self.ewc_lambda * penalty

    def _train_instance(self, images, targets) -> float:
        torch.cuda.empty_cache()
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        _, loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            self._logger.log_message(f"Loss is {loss_value}, stopping training", level=MessageType.ERROR)
            self._logger.log_message(loss_dict)
            sys.exit(1)

        self._optimizer.zero_grad()
        losses.backward()
        self._optimizer.step()

        self._lr_scheduler.step()
        torch.cuda.empty_cache()

        return losses.item()

    @torch.no_grad()
    def _eval_batch(self, images, targets, split_idx, importances=None) -> List[Dict]:
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        detections, _ = self.model(images, targets)

        if importances is not None:
            # Compute EWC Importances
            if split_idx == self._latest_split_id:
                for (k1, p), (k2, imp) in zip(
                        self.model.named_parameters(), importances
                ):
                    assert k1 == k2

                    if p.grad is not None:
                        imp += p.grad.data.clone().pow(2)

        ret = [
            dict(
                x_shape=i.shape,
                pred_bbox=d["boxes"],
                pred_labels=d["labels"],
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
        torch.cuda.empty_cache()
        return ret

    def _train_epoch(self, split: IODSplit, epoch) -> float:
        self._current_train_split = split.index
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
        match_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.model.eval()

        importances = None
        if split.index == self._latest_split_id:
            importances = zerolike_params_dict(self.model)

        for images, targets in tqdm(match_loader):
            yield self._eval_batch(images, targets, split.index, importances)

        if split.index == self._latest_split_id:
            # average over mini batch length
            for _, imp in importances:
                imp /= float(len(match_loader))

            if self.is_separate or self._latest_split_id == 0:
                self.importances[self._latest_split_id] = importances
            else:
                for (k1, old_imp), (k2, curr_imp) in zip(
                        self.importances[self._latest_split_id - 1], importances
                ):
                    assert k1 == k2, "Error in importance computation."
                    self.importances[self._latest_split_id].append(
                        (k1, (self.decay_factor * old_imp + curr_imp))
                    )

                # clear previous parameter importances
                if self._latest_split_id > 0 and (not self.is_separate):
                    del self.importances[self._latest_split_id - 1]

            self.saved_params[self._latest_split_id] = copy_params_dict(self.model)
            # clear previous parameter values
            if self._latest_split_id > 0 and (not self.is_separate):
                del self.saved_params[self._latest_split_id - 1]
