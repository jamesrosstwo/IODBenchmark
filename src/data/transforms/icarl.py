import numpy as np
import torch

from data.transforms.base import IODTransform
from strategies.context_transformer.data import preproc


class ICaRLTrainTransform(IODTransform):
    @property
    def needs_targets(self):
        return True

    def __init__(self, domain: int = 1):
        super().__init__(domain)
        # self._p = preproc(min_dim, rgb_means, p)

    def __call__(self, image, targets=None):
        if targets is None:
            return image
        target_tensor = torch.hstack([targets["boxes"], targets["labels"].unsqueeze(1)])
        i, t = np.array(image), target_tensor.detach().cpu().numpy()

        # in order to be compatible with mixup
        weight = np.ones((t.shape[0], 1))
        t = np.hstack((t, weight))

        return torch.from_numpy(i), torch.from_numpy(t)


class ICaRLEvalTransform(IODTransform):
    @property
    def needs_targets(self):
        return True

    def __init__(self, domain: int = 2):
        super().__init__(domain)
        # self._p = preproc(min_dim, rgb_means, p)

    def __call__(self, image, targets=None):
        if targets is None:
            return image
        target_tensor = torch.hstack([targets["boxes"], targets["labels"].unsqueeze(1)])
        i, t = np.array(image), target_tensor.detach().cpu().numpy()

        # in order to be compatible with mixup
        weight = np.ones((t.shape[0], 1))
        t = np.hstack((t, weight))

        return torch.from_numpy(i), torch.from_numpy(t)
