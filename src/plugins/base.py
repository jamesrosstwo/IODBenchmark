from typing import List, TYPE_CHECKING

from pyhocon import ConfigTree

from data.dataset import IODSplit, IODDataset
from experiments.result import ExperimentResult

if TYPE_CHECKING:
    from experiments.experiment import IODExperiment

from utils.general import find_type


class IODPlugin:
    def __init__(self, name, exp: "IODExperiment", *args, **kwargs):
        self.name: str = name
        self._exp: "IODExperiment" = exp
        self._device = self._exp.torch_device

    """
    Event functions to be implemented by extending plugins.
    """

    def before_dataset(self, dataset: IODDataset):
        pass

    def before_train_split(self, split: IODSplit):
        pass

    def after_train_split(self, split: IODSplit, losses: List[float]):
        pass

    def before_eval_split(self, split: IODSplit):
        pass

    def after_eval_split(self, split: IODSplit):
        pass

    def after_dataset(self, dataset: IODDataset, res: ExperimentResult):
        pass

    @classmethod
    def from_conf(cls, name: str, exp: "IODExperiment", conf: ConfigTree):
        if "cls" not in conf:
            msg = "cls not present in {0} config. Ensure that a class is specified for all plugins".format(name)
            raise KeyError(msg)

        c = conf.get_string("cls")
        conf_d = conf.as_plain_ordered_dict()
        del conf_d["cls"]
        return find_type(c)(name, exp, **conf_d)
