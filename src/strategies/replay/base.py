import math
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Generator

import numpy as np
import torch
import torch.distributed as dist
import plotly.express as px
from pyhocon import ConfigTree
from torchvision.transforms import transforms
from tqdm import tqdm
import pandas as pd

from data.replay import ReplaySplit
from loggers.base import MessageType
from models.faster_rcnn import IODFasterRCNN
from plugins.split_plotting import get_label_occurrence
from src.data.dataset import IODDataset, IODSplit
from src.metrics.base import IODMetric
from src.strategies.base import IODStrategy
from src.utils.general import collate_fn, find_type
import pickle
from typing import TYPE_CHECKING

from utils.memory import MemoryBuffer

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


def labels_to_cls_idx(target):
    return target["labels"][torch.randperm(len(target["labels"]))[0]].item()


class ReplayStrategy(IODStrategy, ABC):
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
        self._replay_epochs = cfg.get_int("replay_epochs", 1)
        self._transforms = [transforms.ToTensor()]
        # Dict of {class_id: List[Datapoint]}
        self._checkpoint_interval = cfg.get_int("checkpoint_interval", 10)
        self._should_plot_memory = cfg.get_bool("plot_memory", False)
        self._buf_size = cfg.get_int("buffer_size", self._dataset.n_splits * 10)
        self._buffer: MemoryBuffer = self._initialize_buffer()
        self._latest_replay_split: ReplaySplit = ReplaySplit.empty(self._dataset.train_splits[0])

    @property
    def buf_size(self):
        """
        Override to customize buffer size in other strategies
        """
        return self._buf_size

    def _initialize_buffer(self) -> MemoryBuffer:
        buffer = self._cfg.get_config("buffer").copy()
        buffer_cls_name = buffer["cls"]
        del buffer["cls"]
        buffer_cls = find_type(buffer_cls_name)
        b = buffer_cls(self.buf_size, self._device, **buffer)
        return b

    def _construct_exemplar_split(self, template_split: IODSplit) -> ReplaySplit:
        """
        Constructs a ReplaySplit from the pre-existing buffer
        """
        rs = ReplaySplit.from_buffer(self._buffer, template_split)
        self._latest_replay_split = rs
        return rs

    @abstractmethod
    def _update_buffer(self, split: IODSplit):
        """
        Add new exemplars to self._buffer
        :param split: New exemplars are selected from the current split
        """
        raise NotImplementedError()

    def _train_instance(self, images, targets):
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

        loss_dict["loss"] = losses
        update_d = loss_dict
        update_d["lr"] = self._optimizer.param_groups[0]["lr"]

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

    def _train_epoch(self, split: IODSplit, epoch: int, checkpoint=True) -> float:
        if checkpoint and epoch % self._checkpoint_interval == 0 and dist.get_rank() == 0:
            self._exp.checkpoint(name="split{0}_epoch{1}".format(split.index, epoch))
        # TODO: num workers
        data_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)

        self.model.train()
        for images, targets in tqdm(data_loader):
            self._train_instance(images, targets)

    def train_split(self, split: IODSplit, checkpoint=True, epochs=None):
        if epochs is None:
            epochs = self._get_epochs(split.index)
        for epoch in range(epochs):
            self._train_epoch(split, epoch, checkpoint)

    def after_train_split(self, split: IODSplit):
        pass
        # self._update_buffer(split)
        # self.train_split(self._construct_exemplar_split(split), checkpoint=False, epochs=self._replay_epochs)

    @torch.no_grad()
    def eval_split(self, split: IODSplit, metrics: List[IODMetric]) -> Generator[List[Dict], None, None]:
        match_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.model.eval()
        with torch.no_grad():
            for images, targets in tqdm(match_loader):
                yield self._eval_batch(images, targets, split.index)

    @abstractmethod
    def _pack_memory(self) -> Dict:
        """
        Packs the replay memory into a dictionary for checkpointing
        """
        raise NotImplementedError()

    def _unpack_memory(self, memory: Dict):
        """
        Used for restoring from a checkpoint, unpack the memory from a pickle file and store it in this object
        """
        for k, v in memory.items():
            try:
                setattr(self, k, v)
            except AttributeError as e:
                self._logger.log_message("Tried to unpack a memory property that does not exist. "
                                         "Ensure the keys of _pack_memory aligns with the class properties.",
                                         level=MessageType.ERROR)
                raise e

    def checkpoint(self, checkpoint_path: Path):
        model_out_path = checkpoint_path / self.model.__class__.__name__
        model_out_path.mkdir(exist_ok=True)
        self.iod_model.checkpoint(model_out_path)

        with open(str(checkpoint_path / "replay.pickle"), "wb") as replay_file:
            pickle.dump(self._pack_memory(), replay_file)

        if self._should_plot_memory:
            contents_path = checkpoint_path / "Memory"
            contents_path.mkdir(exist_ok=True)
            self.plot_contents(contents_path)

    def restore(self, checkpoint_path: Path):
        self._logger.log_message("Restoring strategy from {0}".format(checkpoint_path))
        self.iod_model.restore(self._exp, checkpoint_path / self.iod_model.__class__.__name__)

    def plot_contents(self, path: Path):
        mem = self._latest_replay_split
        label_hist, label_div = get_label_occurrence(mem, num_classes=self._exp.current_dataset.num_classes)

        norm_hist = label_hist / label_div

        for data in [label_hist, norm_hist]:
            name = "{0}_replay_memory_gpu{1}".format(self.__class__.__name__, dist.get_rank())
            title = "Class Distributions for " + name
            if np.sum(data) > 0:
                data = np.log(np.array(data))
                title = "Log " + title

            data_df = pd.DataFrame.from_dict(
                dict(
                    label=list(self._exp.current_dataset.class_names.values()),
                    count=data
                ),
            )
            fig = px.bar(data_df, x="label", y="count", text_auto=True)

            fig.update_layout(  # customize font and legend orientation & position
                title=title,
                #     xaxis_title="Class Index",
                #     yaxis_title="Split Index",
            )
            fig.write_html(path / (name + ".html"))
            fig.write_image(path / (name + ".png"))

        class_sizes = [[] for _ in range(self._exp.current_dataset.num_classes)]
        aspect_ratios = [[] for _ in range(self._exp.current_dataset.num_classes)]

        for t in mem.targets:
            for box, label in zip(t["boxes"], t["labels"]):
                w = box[2] - box[0]
                h = box[3] - box[1]

                class_sizes[label].append(int(np.sqrt(w * h).item()))

                if h == 0 or w == 0:
                    continue
                ratio = (w / h).item().as_integer_ratio()
                reduced_ratio = max(ratio) // min(ratio)
                if ratio[1] > ratio[0]:
                    reduced_ratio = -reduced_ratio
                # Large negative values: Tall and narrow
                # Large positive values: Short and wide
                aspect_ratios[label].append(reduced_ratio)

        bound = int(10e10)
        min_sz = min([min(x) if len(x) > 0 else -bound for x in class_sizes])
        max_sz = max([max(x) if len(x) > 0 else bound for x in class_sizes])
        min_ar = min([min(x) if len(x) > 0 else -bound for x in aspect_ratios])
        max_ar = max([max(x) if len(x) > 0 else bound for x in aspect_ratios])
        sz_bins = list(np.arange(min_sz, max_sz, (max_sz - min_sz) / 10)) + [max_sz]
        ar_bins = list(np.arange(min_ar, max_ar, (max_ar - min_ar) / 10)) + [max_ar]
        class_size_bins = np.stack([np.histogram(x, bins=sz_bins)[0] for x in class_sizes]).T
        ratio_bins = np.stack([np.histogram(x, bins=ar_bins)[0] for x in aspect_ratios]).T

        class_size_fig = px.imshow(class_size_bins, text_auto=True)
        class_size_fig.update_layout(  # customize font and legend orientation & position
            title="Box sizes of instances in replay memory",
            xaxis_title="Class Index",
            yaxis_title="Size Bin",
        )

        class_size_fig.write_html(path / "replay_memory_box_sizes_gpu{0}.html".format(dist.get_rank()))
        class_size_fig.write_image(path / "replay_memory_box_sizes_gpu{0}.png".format(dist.get_rank()))

        aspect_ratio_fig = px.imshow(ratio_bins, text_auto=True)
        aspect_ratio_fig.update_layout(  # customize font and legend orientation & position
            title="Aspect Ratios of instances in replay memory",
            xaxis_title="Class Index",
            yaxis_title="Size Bin",
        )

        aspect_ratio_fig.write_html(path / ("replay_memory_ratios_gpu{0}.html".format(dist.get_rank())))
        aspect_ratio_fig.write_image(path / ("replay_memory_ratios_gpu{0}.png".format(dist.get_rank())))
