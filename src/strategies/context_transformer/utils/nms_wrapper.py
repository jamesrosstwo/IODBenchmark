# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)
import numpy
import pyximport

from definitions import ROOT_PATH

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension


i_dirs = [
    numpy.get_include(),
    str(ROOT_PATH / "src/strategies/context_transformer/utils/nms")
]

script_args = ["--verbose", "--cython-cplus"]

pyximport.install(setup_args={"script_args": script_args, "include_dirs": i_dirs})
from strategies.context_transformer.utils.nms.cpu_nms import cpu_nms
from strategies.context_transformer.utils.nms.gpu_nms import gpu_nms


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if force_cpu:
        # return cpu_soft_nms(dets, thresh, method = 0)
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
