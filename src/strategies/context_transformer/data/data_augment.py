"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import torch
import cv2
import numpy as np
import random
import math

__all__ = ['preproc_for_test', 'preproc', 'BaseTransform']

from strategies.context_transformer.utils.box_utils import matrix_iou


def _crop(image, boxes, labels, cls=None):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale*scale)  # aspect radio
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)
            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])  # roi[1, 4]
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)                               # filter out those centers that are out of the roi area
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if cls is None:
                if len(boxes_t) == 0:
                    continue
            else:
                if len(boxes_t) == 0 or (labels_t != (cls+1)).all():
                    continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    while True:
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))  # [0.5, 2]
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets, cls=None):
        # TODO: This whole function could be converted to torch
        image = np.array(image).astype("float32")
        targets = targets.detach().cpu().numpy().astype("float32")
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # TODO can share the procedure of convert the box to percent format
        # convert to percent format, o means origin image
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o))


        if cls is None:
            image_t, boxes, labels = _crop(image, boxes, labels)
        else:
            image_t, boxes, labels = _crop(image, boxes, labels, cls)

        image_t = _distort(image_t)  # image channels undergo linear transform, also change the saturability

        image_t, boxes = _expand(image_t, boxes, self.means, self.p)

        image_t, boxes = _mirror(image_t, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if cls is None:
            if len(boxes_t) == 0:
                image = preproc_for_test(image_o, self.resize, self.means)
                return image, targets_o
        else:
            if len(boxes_t) == 0 or (labels_t != (cls + 1)).all():
                image = preproc_for_test(image_o, self.resize, self.means)
                return image, targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        _transforms (_transforms) : callable _transforms to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    # TODO: Remove inefficient transfer between gpu and cpu
    def __call__(self, img, target=None):
        img = img.detach().cpu().numpy().astype("float32")
        img = np.moveaxis(img, 0, -1)
        if target is not None:
            height, width, _ = img.shape
            boxes = target[:, :-1]
            labels = target[:, -1]
            boxes[:, 0::2] /= width
            boxes[:, 1::2] /= height
            labels = np.expand_dims(labels, 1)
            targets = np.hstack((boxes, labels))

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)

        if target is not None:
            return torch.from_numpy(img), targets
        else:
            return torch.from_numpy(img)
