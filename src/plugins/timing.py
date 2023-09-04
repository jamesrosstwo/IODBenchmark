import operator
import random
from typing import List, Callable, Any

import numpy as np
import torch
from torch import nn

from data.dataset import IODSplit, IODDataset
from experiments.experiment import IODExperiment
from plugins.base import IODPlugin


class TimingPlugin(IODPlugin):
    """
    Injects timing functionality into various regions in the experiment pipeline.
    WARNING: Synchronization is required to accurately estimate time usage, so this plugin will significantly
    slow down the experiment pipeline, and should only be used for debugging purposes.
    """

    def __init__(self, name, exp: IODExperiment, measurement_probability: float = 1):
        super().__init__(name, exp)
        self.usage = []
        # Estimate the timing with a only subset of function calls being timed
        self._measurement_probability: float = measurement_probability
        self._time_cache = dict()

    @property
    def _active_model(self):
        return self._exp.strategy.iod_model

    def _time_action(self, action: Callable, cached_name: str, *action_args, **action_kwargs) -> Any:
        if random.uniform(0, 1) > self._measurement_probability:
            return action(*action_args, **action_kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = action(*action_args, **action_kwargs)
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end)
        self._time_cache[cached_name].append(time_ms)
        return res

    def _register_timed_fn(self, action: Callable, cached_name):
        self._time_cache[cached_name] = list()

        def t_fn(*args, **kwargs):
            return self._time_action(action, cached_name, *args, **kwargs)

        return t_fn

    def _log_results(self):
        """
        Logs the mean of all timed results in the cache and resets the storage
        """

        times = {"{0}_time_ms".format(k): np.mean(v) for k, v in self._time_cache.items()}

        self._exp.logger.log(dict(sorted(times.items(), key=operator.itemgetter(1), reverse=True)))
        for k in self._time_cache.keys():
            self._time_cache[k] = list()

    def _tree_hook(self, module: nn.Module, prefix):
        module.forward = self._register_timed_fn(module.forward, cached_name="{0}_fwd".format(prefix))
        for sname, submodule in module._modules.items():
            self._tree_hook(submodule, prefix="{0}.{1}".format(prefix, sname))


    def before_dataset(self, dataset: IODDataset):
        # Swap out the forward function to add timing and logging
        module_name = self._active_model.__class__.__name__
        self._tree_hook(self._active_model, module_name)


    def after_eval_split(self, split: IODSplit):
        self._log_results()

    def after_train_split(self, split: IODSplit, losses: List[float]):
        self._log_results()
