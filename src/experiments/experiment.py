import os
import pickle
import time
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Tuple, Generator, Optional

import numpy as np
import torch
import torch.distributed as dist
from pyhocon import ConfigFactory, HOCONConverter
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch import nn

from experiments.result import ExperimentResult
from plugins.base import IODPlugin
from src.data.dataset import IODDataset, IODSplit
from src.loggers.base import IODLogger
from src.metrics.base import IODMetric
from src.metrics.continual.base import ContinualMetric
from src.metrics.final.base import FinalMetric
from src.utils.general import find_type, nanmean
from strategies.base import IODStrategy


@dataclass
class ExperimentDatasets:
    datasets: List[IODDataset]
    current_dataset_idx: int = 0
    current_train_split_idx: int = 0

    @property
    def current_dataset(self) -> IODDataset:
        return self.datasets[self.current_dataset_idx]

    @property
    def current_train_split(self) -> IODSplit:
        return self.current_dataset.train_splits[self.current_train_split_idx]


class IODExperiment:
    def _load_datasets(self) -> ExperimentDatasets:
        dtst_cfgs = self.conf.get_config("data.datasets")
        dtsts = [IODDataset(cfg=v, name=k) for k, v in dtst_cfgs.items()]
        return ExperimentDatasets(dtsts)

    def _make_strategy(self, dtst: IODDataset) -> IODStrategy:
        cfg = self.conf.get_config("strategy")
        strategy_str = cfg.get_string("strategy")
        strategy_type: Type[IODStrategy] = find_type(strategy_str)
        s: IODStrategy = strategy_type(cfg, dtst, self)
        if self.torch_device.type == "cuda" and dist.get_world_size() > 1:
            s.model = nn.SyncBatchNorm.convert_sync_batchnorm(s.model)
            s.model = DistributedDataParallel(s.model, device_ids=[dist.get_rank()], output_device=dist.get_rank())
        if not self._is_new_exp:
            s.restore(self._restore_path / s.__class__.__name__)
        return s

    def _make_logger(self):
        cfg = self.conf.get_config("logger").copy()
        logger_str = cfg.get_string("logger")
        logger_type: Type[IODLogger] = find_type(logger_str)
        del cfg["logger"]
        return logger_type(self, **cfg)

    def _make_metrics(self) -> Tuple[List[ContinualMetric], List[FinalMetric]]:
        cfg = self.conf.get_config("metrics")
        cont_metrics = []
        final_metrics = []
        for k, v in cfg.get_config("continual").items():
            cont_metrics.append(IODMetric.from_conf(k, v, self))

        for k, v in cfg.get_config("final").items():
            final_metrics.append(IODMetric.from_conf(k, v, self))
        return cont_metrics, final_metrics

    def _make_plugins(self):
        from plugins.base import IODPlugin
        plugins = []
        for k, v in self.conf.get_config("plugins").items():
            plugins.append(IODPlugin.from_conf(k, self, v))
        return plugins

    def _setup_ddp(self, rank):
        self.torch_device = torch.device(rank)
        ddp_conf = self.conf.get_config("DDP")
        os.environ['MASTER_ADDR'] = ddp_conf.get_string("master_addr", default="localhost")
        os.environ['MASTER_PORT'] = str(ddp_conf.get("master_port", "12355"))

        backend = str(ddp_conf.get("backend", default="nccl"))

        # initialize the process group
        init_process_group(backend, rank=rank, world_size=ddp_conf.get_int("n_gpus"))

    def _cleanup_ddp(self):
        destroy_process_group()

    @property
    def current_dataset(self) -> IODDataset:
        return self.datasets.datasets[self.datasets.current_dataset_idx]

    def _shared_init(self, rank):
        # mp.set_sharing_strategy('file_system')
        self.torch_device = torch.device(self.conf["torch_device"])
        self.strategy: IODStrategy = None
        metrics = self._make_metrics()
        self._continual_metrics: List[ContinualMetric] = metrics[0]
        self._final_metrics: List[FinalMetric] = metrics[1]

        # If false, only evaluate splits [0, k] after training split k
        self._eval_forward = self.conf.get_bool("eval_forward", True)
        # If true, eval after initialization before any training occurs
        self._eval_init = self.conf.get_bool("eval_init", False)
        self.out_path = Path(self.conf["out_root"]) / self.name
        self._setup_ddp(rank)
        self._checkpoint_path = self.out_path / "checkpoints"
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logger: IODLogger = self._make_logger()
        self._checkpoint_count = 0
        self._eval_train_splits = self.conf.get_bool("eval_train_splits", False)
        self._current_train_split_idx = 0
        self._seen_classes = set()

    @property
    def _is_new_exp(self) -> bool:
        return self._restore_path is None

    def __init__(self, rank, conf, restore_path: Optional[Path] = None, name=None):
        if name is None:
            name = "exp_{0}".format(str(int(time.time())))
        self.name = name
        self.conf = conf
        if restore_path is not None:
            if not restore_path.exists() or not restore_path.is_dir():
                raise FileNotFoundError("Could not find a checkpoint folder at {0}".format(restore_path))
            self.conf = ConfigFactory.parse_file(restore_path / "experiment.conf")
            self._shared_init(rank)
            self.datasets, self._plugins = self._restore(restore_path)
            self._plugins = self._make_plugins()
        else:
            self._shared_init(rank)
            self.datasets: ExperimentDatasets = self._load_datasets()
            self._plugins = self._make_plugins()

        self._restore_path = restore_path

        self._seed = self.conf.get_int("seed", 1234)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(0)

    def _log_cont_res(self, cont_res, prefix=""):
        for metric, d in zip(self._continual_metrics, cont_res):
            self.logger.log_message(
                "{0} Split {1}, {2} eval results Prev: {3}, Current: {4}, Both: {5}, All: {6}".format(
                    prefix,
                    self._current_train_split_idx,
                    metric.name, d["previous"], d["current"], d["both"], d["all"])
            )

    def _eval_splits(self, eval_splits, dataset=None, eval_train=False) -> Tuple[np.ndarray, List]:
        class_occs = np.zeros(self.current_dataset.num_classes)
        for e in eval_splits:
            for k, v in e.class_occurrences.items():
                class_occs[k] += v

        for metric in self._continual_metrics:
            metric.reset()
        if dataset is None:
            dataset = self.current_dataset
        out_results = np.zeros((len(self._continual_metrics), dataset.num_classes, len(eval_splits)))
        for idx, eval_split in enumerate(eval_splits):
            self.logger.log_message("Evaluating split {0}".format(idx))
            if not eval_train:
                self.strategy.before_eval_split(eval_split)
                if dist.get_rank() == 0:
                    for plugin in self._plugins:
                        plugin.before_eval_split(eval_split)
            for res in self.strategy.eval_split(eval_split,
                                                self._continual_metrics):
                for metric in self._continual_metrics:
                    metric.update(res)
            if not eval_train:
                self.strategy.after_eval_split(eval_split)
                if dist.get_rank() == 0:
                    for plugin in self._plugins:
                        plugin.after_eval_split(eval_split)



        agg_results = []
        for i, metric in enumerate(self._continual_metrics):
            # Get continual results for all classes and evaluation splits and store it in the ndarray
            class_map = None
            if eval_train:
                class_map = {self._current_train_split_idx: 0}
            res = metric.result(len(eval_splits), dataset.num_classes).detach().cpu().numpy()
            out_results[i] = res
            self.logger.log_cont_result(metric, res)

            idx = self._current_train_split_idx
            c_class = self.current_dataset.split_classes[idx]
            b_class = set(c_class).union(self._seen_classes)
            p_res = nanmean(res[list(self._seen_classes), :idx + 1], weights=class_occs[list(self._seen_classes)])
            c_res = nanmean(res[c_class, idx], weights=class_occs[c_class])
            b_res = nanmean(res[list(b_class), idx], weights=class_occs[list(b_class)])
            # Omit background
            a_res = nanmean(res[1:, idx], weights=class_occs)

            metric.reset()

            agg_results.append({"previous": p_res, "current": c_res, "both": b_res, "all": a_res})
        return out_results, agg_results

    def run(self) -> Generator[ExperimentResult, None, None]:
        self.logger.log_message("Running experiments \"{0}\"".format(self.name))
        self.logger.log_message("Config Used: {0}".format(self.conf))

        dtst_start = self.datasets.current_dataset_idx
        split_start = self.datasets.current_train_split_idx
        for dataset in self.datasets.datasets[dtst_start:]:
            starting_time = int(time.time())
            e_name = "{0}_{1}".format(self.name, dataset.name)
            self.strategy = self._make_strategy(dataset)

            # Keep track of each class label and how long it has been since it last appeared.
            label_last_seen = np.empty((dataset.n_splits, dataset.num_classes))
            # We have not seen any classes, so we need a placeholder
            label_last_seen[:] = np.nan
            label_occurrences = np.zeros(dataset.num_classes, dtype=np.int)
            self.strategy.before_dataset(dataset)
            if dist.get_rank() == 0:
                for plugin in self._plugins:
                    plugin.before_dataset(dataset)

            self.logger.log_message(
                "Running strategy {0} on dataset \"{1}\"".format(self.strategy, dataset.name))

            # Metric, Class, Split trained until, Split evaluated on
            continual_results = np.zeros(
                (len(self._continual_metrics), dataset.num_classes, dataset.n_splits, dataset.n_splits))
            self._seen_classes = set()

            if self._eval_init:
                _, agg_eval_res = self._eval_splits(dataset.test_splits)
                del _
                self._log_cont_res(agg_eval_res, prefix="Eval")

            for train_split in dataset.train_splits[split_start:]:
                self._current_train_split_idx = train_split.index
                self.logger.log_message("Training on split {0}".format(train_split.index))

                self.strategy.before_train_split(train_split)
                if dist.get_rank() == 0:
                    for plugin in self._plugins:
                        plugin.before_train_split(train_split)

                # Update Label Occurrences
                for t in train_split.targets:
                    label_last_seen[train_split.index, :] += 1

                    seen_labels = set(t["labels"].tolist())
                    for l in seen_labels:
                        label_last_seen[train_split.index, l] = 0
                        label_occurrences[l] += 1

                losses = self.strategy.train_split(train_split)
                self.strategy.after_train_split(train_split)
                if dist.get_rank() == 0:
                    for plugin in self._plugins:
                        plugin.after_train_split(train_split, losses)

                splits_to_eval = dataset.test_splits
                if not self._eval_forward:
                    splits_to_eval = dataset.test_splits[:train_split.index + 1]
                t_idx = self._current_train_split_idx
                continual_results[:, :, t_idx, :], agg_res = self._eval_splits(splits_to_eval)

                if self._eval_train_splits:
                    _, agg_train_res = self._eval_splits(dataset.train_splits)
                    del _
                    self._log_cont_res(agg_train_res, "Train")

                cont_results = {
                    self._continual_metrics[i].name: continual_results[i] for i in range(len(continual_results))
                }

                final_results = {}

                for final_metric in self._final_metrics:
                    tracked_results = {k: v for k, v in cont_results.items() if k in final_metric.tracked_metrics}
                    final_results[final_metric.name] = final_metric.result(
                        dataset,
                        label_last_seen,
                        label_occurrences,
                        **tracked_results
                    )

                if dist.get_rank() == 0:
                    self.logger.log_message("final results: {0}".format(final_results))

                clss = self.current_dataset.split_classes
                self._seen_classes = self._seen_classes.union(set(clss[self._current_train_split_idx]))

            split_start = 0

            cont_results = {
                self._continual_metrics[i].name: continual_results[i] for i in range(len(continual_results))
            }
            final_results = {}

            for final_metric in self._final_metrics:
                tracked_results = {k: v for k, v in cont_results.items() if k in final_metric.tracked_metrics}
                final_results[final_metric.name] = final_metric.result(
                    dataset,
                    label_last_seen,
                    label_occurrences,
                    **tracked_results
                )

            if dist.get_rank() == 0:
                self.logger.log_message("final results: {0}".format(final_results))
            ending_time = int(time.time())
            self.logger.log_message(
                "GPU {0} Completed experiments {1} in {2} seconds.".format(dist.get_rank(), e_name,
                                                                           ending_time - starting_time))

            res = ExperimentResult(name=e_name, continual_results=cont_results, final_results=final_results)
            self.strategy.after_dataset(dataset)
            if dist.get_rank() == 0:
                for plugin in self._plugins:
                    plugin.after_dataset(dataset, res)

            if dist.get_rank() == 0:
                self.logger.log_message("IODExperiment results: {0}".format(res))

            yield res
            self.datasets.current_dataset_idx += 1

    def _restore(self, path: Path) -> Tuple[ExperimentDatasets, List[IODPlugin]]:
        with open(str(path / "datasets.pkl"), "rb") as d_f:  # , open(str(path / "plugins.pkl"), "rb") as p_f:
            d: ExperimentDatasets = pickle.load(d_f)
            # p: List[IODPlugin] = pickle.load(p_f)
            return d, None

    """
    IMPORTANT: To give the user more control, deciding when to checkpoint (and the names of these checkpoints) 
    are left to the strategies.
    """

    def checkpoint(self, name: str, force=False):
        if force or dist.get_rank() != 0:
            return
        strategy = self.strategy
        if name is None:
            name = "checkpoint{0}".format(self._checkpoint_count)

        checkpoint_path = self._checkpoint_path / "{0}_gpu{1}".format(name, dist.get_rank())
        checkpoint_path.mkdir()

        with open(str(checkpoint_path / "experiment.conf"), "w") as conf_file:
            conf_file.write(HOCONConverter.convert(self.conf, "hocon"))

        with open(str(checkpoint_path / "datasets.pkl"), "wb") as d_f:
            pickle.dump(self.datasets, d_f)

        # with open(str(checkpoint_path / "plugins.pkl"), "wb") as p_f:
        #     try:
        #         pickle.dump(self._plugins[0], p_f)
        #     except AttributeError as e:
        #         raise e

        strat_path = checkpoint_path / strategy.__class__.__name__
        strat_path.mkdir(exist_ok=True)
        strategy.checkpoint(strat_path)
        self._checkpoint_count += 1

    def __del__(self):
        self._cleanup_ddp()
