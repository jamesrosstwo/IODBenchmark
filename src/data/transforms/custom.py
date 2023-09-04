from data.transforms.base import IODTransform
from utils.general import find_type


class CustomClassTransform(IODTransform):
    def __init__(self, callable_cls: str, needs_targets: bool, domain: int = 0, **kwargs):
        super().__init__(domain)
        c = find_type(callable_cls)
        self._needs_targets = needs_targets
        self._t = c(**kwargs)

    def __call__(self, image, targets=None):
        if self._needs_targets:
            return self._t(image, targets)
        return self._t(image)

    @property
    def needs_targets(self):
        return self._needs_targets


class CustomFunctionTransform(IODTransform):
    def __init__(self, fn: str, needs_targets: bool, domain: int = 0):
        super().__init__(domain)
        self._needs_targets = needs_targets
        self._t = find_type(fn)

    def __call__(self, image, targets=None):
        if self._needs_targets:
            return self._t(image, targets)
        return self._t(image)

    @property
    def needs_targets(self):
        return self._needs_targets
