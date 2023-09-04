from torchmetrics.image import KernelInceptionDistance

from experiments.experiment import IODExperiment

import torch
from pyhocon import ConfigTree

from src.data.dataset import IODDataset, IODSplit

from typing import TYPE_CHECKING

from strategies.dists import DistsStrategy, SplitDistribution

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment


class KIDDistribution(SplitDistribution):

    def __init__(self, split: IODSplit, n_exemplars: int):
        super().__init__(split)

        exemplars = self._rand_sample_n(split, n_exemplars)
        all_images = torch.stack(exemplars)

    def dist(self, other: "SplitDistribution") -> float:
        pass


class KIDStrategy(DistsStrategy):
    def __init__(self, cfg: ConfigTree, dataset: IODDataset, exp: "IODExperiment"):
        self._module_confs = cfg.get_config("kernel_inception_distance", default=dict())
        self._dist_exemplars_per_split: int = cfg.get_int("dist_exemplars_per_split", 50)
        super().__init__(cfg, dataset, exp)

    def before_train_split(self, split: IODSplit):
        self._dists[split.index] = KIDDistribution(split, self._dist_exemplars_per_split)

        for match_idx in range(split.index):
            module = KernelInceptionDistance(**self._module_confs)
            module.update((self._subsets[split.index] * 255).byte(), real=True)
            module.update((self._subsets[match_idx] * 255).byte(), real=False)
            self._dist_mx[split.index, match_idx] = module.compute()
