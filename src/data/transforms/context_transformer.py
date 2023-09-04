import numpy as np
import torch

from data.transforms.base import IODTransform
from strategies.context_transformer.data import preproc


class ContextTransformerTransform(IODTransform):
    @property
    def needs_targets(self):
        return True

    def __init__(self, min_dim, rgb_means, p, domain: int = 0):
        super().__init__(domain)
        self._p = preproc(min_dim, rgb_means, p)

    def __call__(self, image, targets=None):
        if targets is None:
            return image
        target_tensor = torch.hstack([targets["boxes"], targets["labels"].unsqueeze(1)])
        i, t = self._p(image, target_tensor)

        # in order to be compatible with mixup
        weight = np.ones((t.shape[0], 1))
        t = np.hstack((t, weight))

        return torch.from_numpy(i), torch.from_numpy(t)
