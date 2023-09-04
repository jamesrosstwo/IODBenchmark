from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from experiments.experiment import IODExperiment
from loggers.base import MessageType
from models.base import IODModel
from utils.torchvision.roi_heads import roi_forward
from utils.torchvision.rpn import rpn_forward


class IODFasterRCNN(IODModel):
    def __init__(self, exp: IODExperiment, num_classes, freeze_backbone=True, freeze_rpn=True, **kwargs):
        super().__init__(exp)

        model_kwargs = dict(
            # box_score_thresh=0.05,
            box_nms_thresh=0.5
        )
        # # transform parameters
        # min_size = 800, max_size = 1333,
        # image_mean = None, image_std = None,
        # # RPN parameters
        # rpn_anchor_generator = None, rpn_head = None,
        # rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        # rpn_nms_thresh = 0.7,
        # rpn_fg_iou_thresh = 0.7, rpn_bg_iou_thresh = 0.3,
        # rpn_batch_size_per_image = 256, rpn_positive_fraction = 0.5,
        # rpn_score_thresh = 0.0,
        # # Box parameters
        # box_roi_pool = None, box_head = None, box_predictor = None,
        # box_score_thresh = 0.05, box_nms_thresh = 0.5, box_detections_per_img = 100,
        # box_fg_iou_thresh = 0.5, box_bg_iou_thresh = 0.5,
        # box_batch_size_per_image = 512, box_positive_fraction = 0.25,
        # bbox_reg_weights = None

        model_kwargs.update(kwargs)
        self._model: FasterRCNN = fasterrcnn_resnet50_fpn(weights="COCO_V1", **model_kwargs).to(self._device)
        self._num_classes = num_classes

        # Replace the classifier with a new one, that has "num_classes" outputs
        # 1) Get number of input features for the classifier
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        # 2) Replace the pre-trained head with a new one

        self._model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, self._num_classes * 4).to(self._device)
        self._model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, self._num_classes).to(self._device)
        # self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes).to(self._device)

        # The standard torchvision ROI heads don't output the class logits. Swap out this forward function
        # for one that will give us the logits we need for metrics
        def roi_fwd(*args, **kwargs):
            return roi_forward(self._model.roi_heads, *args, **kwargs)

        def rpn_fwd(*args, **kwargs):
            return rpn_forward(self._model.rpn, *args, **kwargs)

        self._model.roi_heads.forward = roi_fwd
        self._model.rpn.forward = rpn_fwd
        self._checkpoint_name = "{0}.pth".format(IODFasterRCNN.__name__)

        # self._is_backbone_frozen = freeze_backbone
        # if self._is_backbone_frozen:
        #     self._model.backbone.requires_grad_(False)
        #
        # self._is_rpn_frozen = freeze_rpn
        # if self._is_rpn_frozen:
        #     self._model.rpn.requires_grad_(False)

        self._model.requires_grad_(False)
        self._model.roi_heads.box_predictor.requires_grad_(True)

    def forward(self, patterns, targets=None) -> Tuple[Any, Any]:
        res = self._model(patterns, targets)
        if self._model.training:
            return None, res
        return res, None

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
        model_out_path = out_path / self._checkpoint_name
        torch.save(self.state_dict(), str(model_out_path))

    def restore(self, exp: "IODExperiment", checkpoint_path: Path):
        model_path = checkpoint_path / self._checkpoint_name
        try:
            self.load_state_dict(torch.load(model_path))
        except FileNotFoundError as e:
            self._logger.log_message(
                "No checkpoint found for model {0} at {1}".format(IODFasterRCNN.__name__, model_path),
                level=MessageType.ERROR
            )
            raise e

    def construct_optimizer(self):
        params = [p for p in self._model.parameters() if p.requires_grad]

        return torch.optim.SGD(
            params,
            lr=0.001,
            momentum=0.9,
            nesterov=False,
            weight_decay=0.0001
        )
