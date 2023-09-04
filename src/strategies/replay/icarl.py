import operator
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import torch
from pyhocon import ConfigTree
from tqdm import tqdm

from data.replay import ReplaySplit
from models.base import img_embeddings
from src.data.dataset import IODDataset, IODSplit
from src.utils.general import collate_fn

from typing import TYPE_CHECKING

from strategies.replay.base import ReplayStrategy
from utils.memory import MemoryBuffer

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


def labels_to_cls_idx(target):
    return int(torch.mode(target["labels"]).values.item())


class ICaRLStrategy(ReplayStrategy):
    is_parallelizable = True

    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        super().__init__(cfg, dataset, exp)
        self._exemplar_set: Dict[int, Tuple[torch.Tensor, Dict]] = {idx: [] for idx in range(self._dataset.num_classes)}
        self._exemplars_per_class = cfg.get_int("exemplars_per_class")
        self._classes_seen = set()
        self._old_classes_seen = set()
        # Mean embedding for each class in the dataset.
        self._class_means: torch.Tensor

    """
    TODO: Make iCaRL use the buffers
    """

    def _initialize_buffer(self) -> MemoryBuffer:
        return None

    def _preproc(self, images, targets):
        images = list(image.to(self._device) for image in images)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        it = zip(images, targets)
        filtered_it = [(i, t) for i, t in it if t["labels"].shape[0] > 0]
        if len(filtered_it) == 0:
            return None
        return zip(*filtered_it)

    def before_dataset(self, dataset: IODDataset):
        # Calculate the class means across the entire dataset
        self._class_means = None

        occs = [0 for _ in range(dataset.num_classes)]
        with torch.no_grad():
            for split in dataset.train_splits:
                self._exp.logger.log_message("Calculate class means for split {0}:".format(split.index))
                split_loader = split.get_loader(batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
                for images, targets in tqdm(split_loader):

                    res = self._preproc(images, targets)
                    if res is None:
                        continue
                    images, targets = res
                    # Determine which class the datapoints belong to by choosing random element
                    cls_idx = [labels_to_cls_idx(t) for t in targets]
                    features = img_embeddings(self.model, images, targets)
                    if self._class_means is None:
                        self._class_means = torch.zeros((dataset.num_classes, features.shape[1]))
                        self._class_means[:] = torch.nan

                    for c_idx, f in zip(cls_idx, features):
                        occs[c_idx] += 1
                        if self._class_means[c_idx, :].isnan().any():
                            self._class_means[c_idx, :] = f
                            continue
                        self._class_means[c_idx] += f

        self._class_means = (self._class_means / torch.tensor(occs).unsqueeze(-1)).detach().cpu()

    def _exemplar_running_totals(self, classes: Set[int], exemplar_set):
        # The sum from j = 1 to k - 1 of the feature map on each exemplar
        running_totals = {c: [] for c in classes}
        for class_idx, datapoints in exemplar_set.items():
            for idx, ((pattern, target), score) in enumerate(datapoints):
                target = {k: v.to(self._device) for k, v in target.items()}
                feature = img_embeddings(self.model, [pattern.to(self._device)], [target]).detach().cpu().flatten()
                if idx > 0:
                    feature += running_totals[class_idx][-1]
                running_totals[class_idx].append(feature)
        return running_totals

    def _construct_exemplar_split(self, template_split: IODSplit) -> ReplaySplit:
        datapoints: List[Tuple[torch.Tensor, Dict]] = []
        for class_exemplars in self._exemplar_set.values():
            # Just get the exemplar, not the score
            datapoints += [x[0] for x in class_exemplars]
        return ReplaySplit(datapoints, template_split.root, template_split.transform)

    def _update_buffer(self, split: IODSplit):
        m = self._exemplars_per_class

        # Reduce exemplar set - Algorithm 5
        self._exemplar_set = {k: sorted(v, key=operator.itemgetter(1))[:m] for k, v in self._exemplar_set.items()}
        # Construct exemplar set - Algorithm 4
        self.model.eval()
        new_classes = self._classes_seen - self._old_classes_seen
        with torch.no_grad():
            new_exemplars = defaultdict(list)
            for k in range(0, m):
                self._logger.log_message("Adjusting exemplar sets: Step {0}".format(k))
                running_totals = self._exemplar_running_totals(new_classes, new_exemplars)
                self._logger.log_message(
                    "Running total lengths: " + str({k: len(v) for k, v in running_totals.items()}))
                herding_heurs = defaultdict(list)
                split_loader = split.get_loader(batch_size=self._batch_size, shuffle=False, collate_fn=collate_fn)
                for batch_idx, (images, targets) in tqdm(enumerate(split_loader)):
                    res = self._preproc(images, targets)
                    if res is None:
                        continue
                    images, targets = res
                    feats = img_embeddings(self.model, images, targets)
                    for mini_idx, (img, target) in enumerate(zip(images, targets)):
                        class_idx = labels_to_cls_idx(target)
                        if class_idx not in new_classes:
                            continue

                        if len(running_totals[class_idx]) <= k - 1:
                            continue
                        actual_idx = batch_idx * self._batch_size + mini_idx
                        feat = feats[mini_idx]
                        heuristic_vec = self._class_means[class_idx]
                        if k > 0:
                            heuristic_vec -= (running_totals[class_idx][k - 1] + feat) / (k + 1)
                        else:
                            heuristic_vec -= feat
                        heuristic = torch.norm(heuristic_vec)
                        herding_heurs[class_idx].append((actual_idx, heuristic))

                # Storing the index within the split of the min, as well as the value of the min
                chosen_exemplars = {k: sorted(v, key=operator.itemgetter(1))[0] for k, v in herding_heurs.items()}

                # Storing the exemplars themselves by pulling them from the split
                for e_key, exemplar in chosen_exemplars.items():
                    exemplar_idx = exemplar[0]
                    ex_score = exemplar[1]
                    datapoint = split[exemplar_idx]
                    # For the reduction step we need to store the heuristic as well
                    new_exemplars[e_key].append((datapoint, ex_score.detach().cpu()))
            for cls_idx, exemplars in new_exemplars.items():
                self._exemplar_set[cls_idx] += exemplars

    def before_train_split(self, split: IODSplit):
        self._old_classes_seen = self._classes_seen.copy()
        self._classes_seen = self._classes_seen.union(split.labels)

    def _pack_memory(self) -> Dict:
        """
        Packs the replay memory into a dictionary for checkpointing
        :return:
        """
        return dict(
            _exemplar_set=self._exemplar_set,
            _classes_seen=self._classes_seen,
            _old_classes_seen=self._old_classes_seen,
            _class_means=self._class_means
        )
