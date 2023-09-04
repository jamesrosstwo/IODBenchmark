from typing import Dict, List, Tuple

import numpy as np
import torch

from sklearn.cluster import KMeans

from data.coco import COCOLoader, COCOSplit
import torch.distributed as dist


class COCODisjointLoader(COCOLoader):
    """
    Organize the annotations associated with each image present within the passed instances into a
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._n_splits: int = kwargs["num_splits"]
        # Creates the splits dividing the images by object class
        self._split_indices = self._create_split_indices()

    def _create_split_indices(self) -> Dict[int, List[int]]:

        id_frame_map = {}
        for image in self.instances["images"]:
            id_frame_map[image["id"]] = self._frames_path / image["file_name"]

        box_classes = [[x["category_id"] for x in a] for a in self._annotations.values()]
        img_class_counts = np.array([np.bincount(x, minlength=self._n_classes).astype(np.float64) for x in box_classes])

        # Omit the background class, we don't want to choose our splits based on the occurrence of the background
        # class.
        img_class_counts = img_class_counts[:, 1:]
        eps = np.finfo(float).eps
        img_class_counts = img_class_counts / np.linalg.norm(img_class_counts + eps, axis=1, keepdims=True)

        clusters = KMeans(n_clusters=self._num_splits, random_state=0).fit(img_class_counts)

        split_indices = clusters.labels_
        splits = {x: ([], []) for x in range(self._num_splits)}

        annot_keys = list(self._annotations.keys())
        annot_vals = list(self._annotations.values())
        for idx, split_idx in enumerate(split_indices):
            frame_id = annot_keys[idx]
            splits[split_idx][0].append(id_frame_map[frame_id])
            splits[split_idx][1].append(annot_vals[idx])

        return splits

    def construct_splits(self) -> Tuple[List[COCOSplit], List[COCOSplit], int]:
        train_splits = []
        test_splits = []
        for split_id, split in self._split_indices.items():
            indices = torch.arange(len(split[0]), dtype=torch.int64)

            s_new = list(zip(*split))

            train_sz = int(len(indices) * self._train_test_split)
            train_data = list(zip(*s_new[:train_sz]))
            eval_data = list(zip(*s_new[train_sz:]))

            train_split = COCOSplit(*train_data, root=str(self._root), split_id=split_id, rank=dist.get_rank(),
                                    n_gpus=dist.get_world_size(),
                                    transforms=self._train_ts,
                                    shuffle=True)
            eval_split = COCOSplit(*eval_data, root=str(self._root), split_id=split_id, rank=dist.get_rank(),
                                   n_gpus=dist.get_world_size(),
                                   transforms=self._eval_ts)

            train_splits.append(train_split)
            test_splits.append(eval_split)
        return train_splits, test_splits, len(self._split_indices)
