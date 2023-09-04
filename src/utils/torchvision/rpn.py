from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers
from typing import List, Dict, Tuple

from torch import Tensor

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList


def rpn_forward(
        self: RegionProposalNetwork,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
) -> Tuple[List[Tensor], Dict[str, Tensor]]:
    """
    Args:
        images (ImageList): images for which we want to compute the predictions
        features (Dict[str, Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.

    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[str, Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """
    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
    regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = self.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }
    return boxes, losses
