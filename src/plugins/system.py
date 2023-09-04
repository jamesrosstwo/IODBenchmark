import errno
import os
import sys
from typing import List

import torch

from data.dataset import IODSplit
from experiments.experiment import IODExperiment
from plugins.base import IODPlugin
from utils.debug import get_free_vram


def get_open_files():
    """List process currently open FDs and their target """
    if not sys.platform.startswith('linux'):
        raise NotImplementedError('Unsupported platform: %s' % sys.platform)

    ret = {}
    base = '/proc/self/fd'
    for num in os.listdir(base):
        path = None
        try:
            path = os.readlink(os.path.join(base, num))
        except OSError as err:
            # Last FD is always the "listdir" one (which may be closed)
            if err.errno != errno.ENOENT:
                raise
        ret[int(num)] = path
    return ret


class SystemUsagePlugin(IODPlugin):
    def __init__(self, name, exp: IODExperiment):
        super().__init__(name, exp)
        self.usage = []


    def _get_results(self):
        return {
            "VRAMUsage": get_free_vram(),
            "OpenFiles": len(get_open_files())
        }

    def _log_results(self):
        self._exp.logger.log(self._get_results())

    def before_eval_split(self, split: IODSplit):
        self._log_results()

    def after_eval_split(self, split: IODSplit):
        self._log_results()

    def before_train_split(self, split: IODSplit):
        self._log_results()

    def after_train_split(self, split: IODSplit, losses: List[float]):
        self._log_results()
