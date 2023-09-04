import operator
from collections import defaultdict
from pathlib import Path

import plotly.express as px
import numpy as np
from pyhocon import ConfigTree
from sklearn.mixture import GaussianMixture

from loggers.base import MessageType
from src.data.dataset import IODSplit, IODDataset

from typing import TYPE_CHECKING, Dict

from strategies.replay.base import ReplayStrategy

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class IdiosyncracyStrategy(ReplayStrategy):
    """
    Remember instances from classes that express traits that are idiosyncratic of that class.
    """

    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        self._exemplars_per_class = cfg.get_int("exemplars_per_class", 3)
        self._mixture_args = cfg.get_config("mixture_args", dict())
        assert "n_components" not in self._mixture_args, "n_components is determined by the number of exemplars " \
                                                         "stored per class "

        super().__init__(cfg, dataset, exp)
        self._mixtures = dict()
        n_cls = self._dataset.num_classes
        self._similarity_matrices = np.zeros((self._dataset.n_splits, n_cls, n_cls))
        self._similarity_matrices[:] = np.nan
        self._latest_split_idx = 0

    @property
    def buf_size(self):
        return self._exemplars_per_class * self._dataset.num_classes

    def _update_buffer(self, split: IODSplit):
        n_c = self._dataset.num_classes
        box_classes = [t["labels"].detach().cpu().numpy() for t in split.targets]

        def get_bins(x: np.ndarray):
            if x.shape[0] == 0:
                return np.zeros(n_c, dtype=np.float64)
            return np.bincount(x, minlength=n_c).astype(np.float64)

        img_class_counts = np.stack([get_bins(x) for x in box_classes])
        most_common_class = np.argmax(img_class_counts, axis=1)
        embeddings = defaultdict(list)

        self._logger.log_message("Selecting exemplars for split {0}".format(split.index))
        all_probas = np.empty((self._exemplars_per_class, len(split.labels)))
        for class_idx in split.labels:
            print(class_idx)
            if class_idx in self._mixtures:
                self._logger.log_message("Tried to fit class {0} more than once".format(class_idx),
                                         level=MessageType.WARNING)
                continue
            self._logger.log_message("\tSelecting exemplars for class {0} in split {1}".format(class_idx, split.index))
            class_indices = np.asarray(most_common_class == class_idx).nonzero()[0]

            if class_indices.shape[0] <= 1:
                self._logger.log_message(
                    "Class {0} has {1} exemplars, too few to use with the mixture strategy. Skipped.".format(
                        class_idx,
                        class_indices.shape[0]
                    ),
                    level=MessageType.WARNING
                )
                continue

            for c_i in class_indices:
                pattern, target = split[c_i]
                pattern = pattern.to(self._device)
                target = {k: v.to(self._device) for k, v in target.items()}
                embeddings[class_idx].append(
                    self.iod_model.img_embeddings([pattern], [target]).detach().cpu().numpy().reshape(-1))

            stacked_es = np.vstack(embeddings[class_idx])
            n_comps = min(self._exemplars_per_class, stacked_es.shape[0])
            self._mixtures[class_idx] = GaussianMixture(n_components=n_comps, **self._mixture_args)
            self._mixtures[class_idx] = self._mixtures[class_idx].fit(stacked_es)

            all_probas[:, class_idx] = self._mixtures[class_idx].predict_proba(stacked_es)
            exemplar_indices = np.argmax(probs, axis=0)
            self._buffer.push([split[i] for i in exemplar_indices])

            # indices = list(range(len(embeddings)))
            # sorted_scores = sorted(list(zip(list(scores), indices)), reverse=True, key=operator.itemgetter(0))
            # exemplar_indices = [x[1] for x in sorted_scores[:self._exemplars_per_class]]

        for class_idx in split.labels:
            if len(embeddings[class_idx]) == 0:
                continue
            for mixture_cls_idx, mixture in self._mixtures.items():
                self._similarity_matrices[split.index, mixture_cls_idx, class_idx] = mixture.score(
                    embeddings[class_idx])
        self._latest_split_idx = split.index

    def _pack_memory(self) -> Dict:
        """
        Packs the replay memory into a dictionary for checkpointing
        :return:
        """
        return dict(
            _exemplars_per_class=self._exemplars_per_class,
            _mixture_args=self._mixture_args,
            _mixtures=self._mixtures,
            _dist_matrices=self._similarity_matrices
        )

    def plot_contents(self, path: Path):

        similarity_out = path / "MixtureSimilarities"
        similarity_out.mkdir(exist_ok=True)
        super().plot_contents(similarity_out)

        sim = self._similarity_matrices[self._latest_split_idx, :, :]
        fig = px.imshow(sim)
        fig.write_html(similarity_out / "similarity_train.html")
        fig.write_image(similarity_out / "similarity_train.png")

        if len(self._mixtures) == 0:
            return

        mean_arr = [x.means_ for k, x in sorted(self._mixtures.items(), key=operator.itemgetter(0))]
        means = np.stack(mean_arr)
        covs = np.stack([x.covariances_ for k, x in sorted(self._mixtures.items(), key=operator.itemgetter(0))])
        keys = sorted(self._mixtures.keys())

        np.savez(str(similarity_out / "gaussians.npz"), means=means, covs=covs, keys=keys)
