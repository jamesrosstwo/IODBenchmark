import collections
import json
from abc import ABC
from pathlib import Path
from typing import List, Dict, Set
import random

import numpy as np
import torch
from torchvision.datasets.folder import default_loader

from data.dataset import IODSplit
from data.split_loader import SplitLoader, _construct_transforms
from data.transforms.base import IODTransform
from utils.general import corner_bbox_from_dim


def _path_to_id(p):
    return int(str(p.stem).split(".")[0].lstrip("0"))


class COCOSplit(IODSplit):
    def make_targets(self, annotations: List[Dict]):
        boxes = []
        areas = []
        labels = []
        crowds = []
        image_id: int = np.nan

        for idx, annotation in enumerate(annotations):
            if idx == 0:
                image_id = annotation["image_id"]
            else:
                assert annotation["image_id"] == image_id

            bbox = annotation["bbox"]

            bbox = list(corner_bbox_from_dim(*bbox))
            assert 0 <= bbox[0] < bbox[2]
            assert 0 <= bbox[1] < bbox[3]
            if abs(bbox[3] - bbox[1]) <= 0.5 or abs(bbox[2] - bbox[0]) <= 0.5:
                continue

            assert annotation["area"] > 0

            l: int = annotation["category_id"]

            # Depending on the split, classes that have not yet been seen should be masked out of the ground truth
            # labels to demonstrate background semantic shift with the introduction of new classes
            if isinstance(self._class_mask, list) and l not in self._class_mask:
                del annotations[idx]
                continue

            labels.append(l)
            boxes.append(bbox)
            areas.append(annotation["area"])
            crowds.append(0)

        box_tensor = torch.Tensor(boxes).to(torch.int64)
        label_tensor = torch.Tensor(labels).to(torch.int64)

        assert len(labels) == len(boxes)
        if len(labels) == 0:
            box_tensor = torch.empty((0, 4), dtype=torch.float32)
            label_tensor = torch.empty(0, dtype=torch.int64)

        return {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.Tensor([image_id]).to(torch.int64),
        }

    def __init__(self, frames: List[Path], annotations: List[List[Dict]], root, split_id,
                 class_mask=None,
                 shuffle=False,
                 transforms=None,
                 loader=default_loader):
        super().__init__(split_id, root, None)
        self.__targets = None
        if transforms is None:
            transforms = list()
        self._transforms: List[IODTransform] = transforms
        self.loader = loader

        self._class_mask = None
        if isinstance(class_mask, list):
            self._class_mask = class_mask.copy()
        self.__labels = None

        self.frames = frames
        self._annotations = annotations

        self._class_names = []

        zipped = list(zip(self.frames, self._annotations))
        if shuffle:
            random.shuffle(zipped)

        # Removing datapoints with no objects in the scene will introduce bias.
        # TODO: Look into the cause of removing this causing an error.
        zipped = [x for x in zipped if len(x[1]) > 0]
        self.frames, self._annotations = tuple(zip(*zipped))

    @property
    def targets(self) -> List[Dict[str, torch.Tensor]]:
        if self.__targets is None:
            self.__targets = [self.make_targets(self._annotations[i]) for i in range(len(self))]
        return self.__targets

    @property
    def labels(self) -> Set[int]:
        if self.__labels is None:
            self.__labels = set()
            for t in self.targets:
                self.__labels = self.__labels.union(set(t["labels"].tolist()))
        return self.__labels

    def __getitem__(self, idx):
        target = self.targets[idx]
        img_path = self.frames[idx]
        img = self.loader(str(img_path))

        for t in self._transforms:
            if t.needs_targets:
                img, target = t(img, target)
            else:
                img = t(img)

        return img, target

    def __len__(self):
        return len(self.frames)

    @property
    def class_names(self):
        return []


_COCO_REQS = {"base_path": str, "frame_path": str, "instances_path": str, "train_test_split": float}


class COCOLoader(SplitLoader, ABC):
    def _init_properties(self, kwargs):
        with open(str(self._instances_path), "r") as f:
            self.instances = json.load(f)
        self._id_map = self._construct_id_map()
        self._annotations = self._image_annotations()

    def __init__(self, **kwargs):
        self._validate_cfg(kwargs, _COCO_REQS)
        self._root = Path(kwargs["base_path"])
        self._frames_path: Path = Path(kwargs["frame_path"])
        self._instances_path: Path = Path(kwargs["instances_path"])
        self._train_test_split = kwargs.get("train_test_split", 0.75)
        _t_dict = kwargs.get("transforms", dict())
        self._train_ts, self._eval_ts = _construct_transforms(_t_dict)
        self._frames = list(self._frames_path.rglob("*.jpg"))
        self.__class_names = None

        self._init_properties(kwargs)

    def _image_annotations(self):
        annots = {i["id"]: [] for i in self.instances["images"]}
        for annot in self.instances["annotations"]:
            annot["category_id"] = self._id_map[annot["category_id"]]
            img_id = annot["image_id"]
            if annot["bbox"][2] < 1 or annot["bbox"][3] < 1:
                continue

            annots[img_id].append(annot)
        return collections.OrderedDict(annots)

    def _construct_id_map(self):
        return {x["id"]: idx for idx, x in enumerate(self.instances["categories"])}

    @property
    def class_names(self):
        if self.__class_names is None:
            self.__class_names = {self._id_map[x["id"]]: x["name"] for x in self.instances["categories"]}
            assert 0 not in self.class_names, "Background (class zero) should not be present within the instance labels"
            self.__class_names[0] = "background"
        return self.__class_names

    @property
    def _n_classes(self):
        return len(self._id_map)
