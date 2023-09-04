from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from experiments.experiment import IODExperiment
from models.base import IODModel
from models.deformable_detr import build_model

_DEFAULT_KWARGS = {
    "lr": 2e-4,
    "lr_backbone_names": ["backbone.0"],
    "lr_backbone": 2e-5,
    "lr_linear_proj_names": ['reference_points', 'sampling_offsets'],
    "lr_linear_proj_mult": 0.1,
    "weight_decay": 1e-4,
    "lr_drop": 40,
    "lr_drop_epochs": None,
    "clip_max_norm": 0.1,
    "sgd": False,
    "with_box_refine": False,
    "two_stage": False,
    "frozen_weights": None,
    "backbone": 'resnet50',
    "dilation": False,
    "position_embedding": 'sine',
    "position_embedding_scale": 2 * np.pi,
    "num_feature_levels": 4,
    "enc_layers": 6,
    "dec_layers": 6,
    "dim_feedforward": 1024,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "num_queries": 300,
    "dec_n_points": 4,
    "enc_n_points": 4,
    "masks": False,
    "aux_loss": True,
    "set_cost_class": 2,
    "set_cost_bbox": 5,
    "set_cost_giou": 2,
    "mask_loss_coef": 1,
    "dice_loss_coef": 1,
    "cls_loss_coef": 2,
    "bbox_loss_coef": 5,
    "giou_loss_coef": 2,
    "focal_alpha": 0.25,
    "dataset_file": 'coco',
    "coco_path": './data/coco',
    "remove_difficult": False,
    "output_dir": '',
    "device": 'cuda',
    "seed": 42,
    "resume": '',
    "start_epoch": 0,
    "eval": False,
    "num_workers": 2,
    "cache_mode": False
}


class IODDeformableDETR(IODModel):
    def __init__(self, exp: IODExperiment, **kwargs):
        super().__init__(exp)
        model_args = _DEFAULT_KWARGS.copy()
        model_args.update(kwargs)
        self._model, self._criterion, self._postprocessors = build_model(**model_args)
        self._model = self._model.to(self._device)

    def _preprocess(self, patterns: List[torch.Tensor], targets):
        samples = torch.stack(patterns).to(self._device)
        img_dims = list(samples.shape[2:])
        for t in targets:
            t["orig_size"] = torch.tensor(img_dims).to(self._device)
            t["boxes"] = t["boxes"].to(torch.float32)
            for box_idx, box in enumerate(t["boxes"]):
                box[0] /= float(img_dims[0])
                box[1] /= float(img_dims[1])
                box[2] /= float(img_dims[0])
                box[3] /= img_dims[1]
                t["boxes"][box_idx] = box
        return samples, targets

    def forward(self, patterns, targets=None):
        samples, targets = self._preprocess(patterns, targets)
        outputs = self._model(samples)
        if targets is None:
            return outputs

        # for i, t in enumerate(targets):
        #     for k, v in t.items():
        #         targets[i][k] = v.to(torch.float32)

        loss_dict = self._criterion(outputs, targets)
        weight_dict = self._criterion.weight_dict

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self._postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in self._postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = self._postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        for r in results:
            r["class_logits"] = r["scores"]
            del r["scores"]
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        return results, loss_dict
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)
        #
        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #
        #     panoptic_evaluator.update(res_pano)

    def img_embeddings(self, patterns, targets=None):
        activation: Dict[str, Tensor] = {}

        def get_activation(name):
            def hook(model_hook: Module, x_hook: Tensor, out_hook: Tensor):
                activation[name] = out_hook.detach().cpu()

            return hook

        output_features: List[Tensor] = []

        with torch.no_grad():
            for p, t in zip(patterns, targets):
                with self._model.roi_heads.box_head.register_forward_hook(
                        get_activation("box_head")):
                    self._model([p], [t])

                output_features.append(torch.mean(activation["box_head"], 0))
        return torch.stack(output_features)

    def checkpoint(self, out_path: Path):
        pass

    def restore(self, exp: "IODExperiment", checkpoint_path: Path):
        pass
