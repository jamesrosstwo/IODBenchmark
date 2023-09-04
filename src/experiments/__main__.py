import os
import argparse
from typing import Optional

import torch.multiprocessing as mp
from pathlib import Path
from pyhocon import ConfigFactory

from src.experiments.experiment import IODExperiment


def dir_path(string):
    if not os.path.isdir(string):
        raise NotADirectoryError(string)
    return string


def conf_path(string):
    if not os.path.isfile(string):
        raise Exception("Passed file " + string + " is not a file")

    p = Path(string)
    if not p.exists():
        raise FileNotFoundError()
    if p.suffix != ".conf":
        raise Exception("Passed configuration file was of incorrect type.")

    return string


def checkpoint_path(string):
    if string is None:
        return None
    return Path(string)


def run_ddp(rank, conf, restore_path: Optional[Path]):
    exp = IODExperiment(rank, conf, restore_path)
    for result in exp.run():
        out_path = exp.out_path / "{0}.pickle".format(result.name)
        result.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a continuous learning benchmark.')
    parser.add_argument('-c', '--conf', type=conf_path, required=True)
    parser.add_argument('-r', '--restore', nargs='?', default=None, type=checkpoint_path, required=False)
    args = parser.parse_args()

    restore_path = args.restore

    conf = ConfigFactory.parse_file(args.conf)

    n_gpus = conf.get("DDP.n_gpus", 1)

    mp.spawn(run_ddp,
             args=(conf, restore_path),
             nprocs=n_gpus,
             join=True)
