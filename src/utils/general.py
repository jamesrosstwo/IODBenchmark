import importlib

import numpy as np
import torch
import torch.distributed as dist


def corner_bbox_from_dim(x, y, w, h):
    return x, y, x + w, y + h


def find_type(cls: str):
    m_name = ".".join(cls.split(".")[:-1])
    module = importlib.import_module(m_name)
    return getattr(module, cls.split(".")[-1])


# https://github.com/pytorch/vision/blob/d59292575f1a88b4a129bf4bb75c429021e0eb52/references/detection/utils.py


def collate_fn(batch):
    return tuple(zip(*batch))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the result_keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict


def nanmean(arr: np.ndarray, weights=np.ndarray, **avg_kwargs):
    indices = np.where(np.logical_not(np.isnan(arr)))[0]
    return np.average(arr[indices], weights=weights[indices], **avg_kwargs)
