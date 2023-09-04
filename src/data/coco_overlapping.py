import collections
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.coco import COCOLoader
from data.coco_disjoint import COCOSplit


def _path_to_id(p):
    return int(str(p.stem).split(".")[0].lstrip("0"))


class COCOOverlappingLoader(COCOLoader):
    """
    Splitting COCO into a set of splits where no attempt to minimize the overlap of classes is made.
    Each split is assigned a set of classes, and every datapoint is assigned to a frame by looking at the
    class distribution of labels within that frame. Since each label's class belongs to a certain split, whichever
    split is most common within the labels of an image takes ownership of that datapoint.
    """

    def _init_properties(self, kwargs):
        with open(str(self._instances_path), "r") as f:
            self.instances = json.load(f)

        self._id_map = self._construct_id_map()

        if "split_locations" in kwargs:
            self._n_splits = len(kwargs["split_locations"]) + 1
            self._split_classes = np.array_split(np.array(range(self._n_classes)), kwargs["split_locations"])
        elif "num_splits" in kwargs:
            self._n_splits: int = kwargs["num_splits"]
            self._split_classes = np.array_split(np.array(range(self._n_classes)), self._n_splits)
        elif "splits_path" in kwargs:
            self._n_splits, self._split_classes = self._split_indices_from_path(kwargs["split_path"])
        elif "split_classes" in kwargs:
            self._n_splits = len(kwargs["split_classes"])
            self._split_classes = self._split_indices_from_names(kwargs["split_classes"])
            c_names = [self.class_names[c] for s in self._split_classes for c in s]
            self.instances["categories"] = [x for x in self.instances["categories"] if x["name"] in c_names]
            self._id_map = self._construct_id_map()
            self._split_classes = self._split_indices_from_names(kwargs["split_classes"])
        else:
            raise NameError(
                "Keys `num_splits` or `split_sizes` not passed to COCOOverlapping dataset. Please ensure that the passed config contains all required items.")

        self._annotations = self._image_annotations()

    def __init__(self, exclude_empty=False, **kwargs):
        super().__init__(**kwargs)

        # Creates the splits dividing the images by object class
        self._exclude_empty: bool = exclude_empty
        self._category_split_map = dict()
        self._split_indices = self._create_split_indices()

    def _split_indices_from_names(self, split_class_names):
        # account for bg
        idx_names = {v: k + 1 for k, v in enumerate(self.class_names.values())}
        splits_cnames: List[List[str]] = split_class_names
        out_split_classes = []

        for s_cnames in splits_cnames:
            split_indices = []
            for c in s_cnames:
                assert c in self.class_names.values()
                split_indices.append(idx_names[c])
                del idx_names[c]
            out_split_classes.append(split_indices)

        return out_split_classes

    def _split_indices_from_path(self, splits_loc: str):
        split_path = Path(splits_loc)
        split_files = [x for x in split_path.glob("*/**") if x.is_file()]
        return len(split_files), split_files

    def _construct_id_map(self):
        # Background must remain zero
        d = {0:0}
        d.update({x["id"]: idx + 1 for idx, x in enumerate(self.instances["categories"])})
        return d

    def _image_annotations(self):
        annots = {i["id"]: [] for i in self.instances["images"]}
        keep_idxs = [True for _ in self.instances["annotations"]]
        for i_idx, annot in enumerate(self.instances["annotations"]):
            c = annot["category_id"]
            # Category masking for annotations
            if c not in self._id_map:
                keep_idxs[i_idx] = False
                continue

            annot["category_id"] = self._id_map[annot["category_id"]]
            img_id = annot["image_id"]

            if annot["bbox"][2] < 1 or annot["bbox"][3] < 1:
                keep_idxs[i_idx] = False
                continue

            annots[img_id].append(annot)
        self.instances["annotations"] = [x for idx, x in enumerate(self.instances["annotations"]) if keep_idxs[idx]]

        return collections.OrderedDict(annots)

    def _create_split_indices(self) -> Dict[int, List[int]]:
        id_frame_map = {}
        for image in self.instances["images"]:
            id_frame_map[image["id"]] = self._frames_path / image["file_name"]

        self._category_split_map = {}
        for split_idx, classes in enumerate(self._split_classes):
            for c in classes:
                self._category_split_map[c] = split_idx

        box_splits = []
        for a in self._annotations.values():
            box_split = []
            for x in a:
                c_id = x["category_id"]
                box_split.append(self._category_split_map[c_id])
            box_splits.append(box_split)

        img_split_counts = np.array([np.bincount(x, minlength=self._n_splits).astype(np.float64) for x in box_splits])
        most_common_class = np.argmax(img_split_counts, axis=1)
        instance_mask = [True for _ in box_splits]
        if self._exclude_empty:
            instance_mask = [len(x) > 0 and m for x, m in zip(box_splits, instance_mask)]
        splits = {x: ([], []) for x in range(self._n_splits)}

        annot_keys = list(self._annotations.keys())
        annot_vals = list(self._annotations.values())
        for idx, (mask, split_idx) in enumerate(zip(instance_mask, most_common_class)):
            if not mask:
                continue
            frame_id = annot_keys[idx]
            splits[split_idx][0].append(id_frame_map[frame_id])
            splits[split_idx][1].append(annot_vals[idx])

        return splits

    def construct_splits(self) -> Tuple[List[COCOSplit], List[COCOSplit], int]:
        train_splits = []
        test_splits = []
        seen_classes = set()
        for split_id, split in self._split_indices.items():
            s_new = list(zip(*split))

            # Hacky way of ensuring the splits remain the same across experiments while also ensuring a random
            # distribution of data
            div = 10000
            hashes = [(idx, hash(x) % div) for idx, x in enumerate(split[0])]
            cutoff = self._train_test_split * div
            train_data = list(zip(*[s_new[idx] for idx, x in hashes if x < cutoff]))
            eval_data = list(zip(*[s_new[idx] for idx, x in hashes if x >= cutoff]))
            seen_classes = seen_classes.union(set(self._split_classes[split_id]))

            # We only want to mask the background classes during the training splits.
            # This allows us to still calculate the forward transfer.
            train_split = COCOSplit(*train_data, root=str(self._root), split_id=split_id,
                                    transforms=self._train_ts,
                                    shuffle=True, class_mask=list(seen_classes))
            eval_split = COCOSplit(*eval_data, root=str(self._root), split_id=split_id,
                                   transforms=self._eval_ts)
            train_splits.append(train_split)
            test_splits.append(eval_split)
        return train_splits, test_splits, len(self._split_indices)


    @property
    def split_classes(self):
        return self._split_classes.copy()
