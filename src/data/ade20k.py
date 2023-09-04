import random
from typing import Dict, List, Set

import numpy as np
import torch

from pathlib import Path

from torchvision import transforms
from torchvision.datasets.folder import default_loader

from data.dataset import IODSplit
from data.split_loader import SplitLoader


class ADESplit(IODSplit):
    def make_targets(self, image_id, annotation_path):
        with open(str(annotation_path), "r") as annotation_file:
            return {
                "boxes": torch.Tensor(boxes).to(torch.int64),
                "labels": torch.Tensor(labels).to(torch.int64),
                "image_id": torch.Tensor([image_id]).to(torch.int64),
                "step": step
            }

    def mask_to_bboxes(self, mask):
        raise NotImplementedError()

    def __init__(self, frames: List[Path], annotations: List[Path], root, id, shuffle=False, transform=None,
                 loader=default_loader):
        super().__init__(id, root, transform)
        self.transform = transform
        self.loader = loader

        self._label_map = {}
        self._highest_unseen_label = 0

        self.frames = frames
        self.annotations = annotations
        self.targets = [self.make_targets(i, self.annotations[i]) for i in range(len(self))]

        zipped = list(zip(self.frames, self.annotations, self.targets))
        if shuffle:
            random.shuffle(zipped)

        # Removing datapoints with no objects in the scene will introduce bias.
        # TODO: Look into the cause of removing this causing an error.
        zipped = [x for x in zipped if len(x[2]["labels"]) > 0]
        self.frames, self.annotations, self.targets = tuple(zip(*zipped))

        self.steps = torch.unique(torch.stack([x["step"] for x in self.targets]))

    @property
    def labels(self) -> Set:
        out_labels = set()
        for t in self.targets:
            out_labels = out_labels.union(set(t["labels"].tolist()))

        return out_labels

    def __getitem__(self, idx):
        target = self.targets[idx]
        img_path = self.frames[idx]
        img = self.loader(str(img_path))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.frames)


class ADELoader(SplitLoader):
    def _load_scene_categories(self, file):
        pass

    def __init__(self, root: Path, **kwargs):
        self._root = root
        self._frames_path: Path = Path(root) / "images/train"
        self._annotations_path: Path = Path(root) / "annotations/train"
        self._obj_info_path: Path = Path(root) / "objectInfo150.txt"
        self._categories_path: Path = Path(root) / "sceneCategories.txt"
        with open(self._categories_path, "r") as f:
            self._scene_categories = self._load_scene_categories(f)

        self._frames = list(self._frames_path.rglob("*.jpg"))
        self._annotations = list(self._annotations_path.rglob("*.png"))
        self._steps = np.unique([int(x.parent.stem.split("_")[-1]) for x in self._annotations])

        self._num_splits: int = kwargs["num_splits"]

        # Maps each scenario within the dataset to a split
        self._split_map = self._create_split_map()
        self._splits = self._create_splits(self._split_map)

    def _create_split_map(self) -> Dict[int, int]:
        task_splits = np.array_split(self._steps, self._num_splits)

        e_map = {}
        for split_idx, split in enumerate(task_splits):
            for task_idx in split:
                e_map[int(task_idx.item())] = split_idx

        return e_map

    def _create_splits(self, e_map: Dict[int, int]) -> Dict[int, List[int]]:
        splits = {v: [] for v in e_map.values()}
        for frame, annot in zip(self._frames, self._annotations):
            step = int(annot.parent.stem.split("_")[-1])
            split_id = e_map[step]
            splits[split_id].append((frame, annot))

        for split_id in splits.keys():
            splits[split_id] = list(zip(*splits[split_id]))
        return splits

    def construct_splits(self) -> List[ADESplit]:
        out_splits = []
        for split_id, split in self._splits.items():
            s = ADESplit(*split, root=str(self._root), id=split_id, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
            out_splits.append(s)
        return out_splits
