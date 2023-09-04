from typing import Collection, Tuple

import cv2
import torch
from torchvision import transforms

from utils.memory import MemoryBuffer


class JPEGBuffer(MemoryBuffer):
    """
    A queue to replay instances during training.
    """

    def __init__(self, capacity: int, device: torch.device, quality: int = 90):
        super().__init__(capacity, device)
        self._quality = quality
        self._to_tensor = transforms.ToTensor()

    def push_single(self, entry: Tuple, quality_multiplier: float = 1):
        if self._memory.full():
            self._memory.get()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._quality * quality_multiplier]

        cv2_img = entry[0].permute(1, 2, 0).detach().cpu().numpy()
        result, encimg = cv2.imencode('.jpg', cv2_img, encode_param)

        self._memory.put_nowait((encimg, entry[1]))

    def push(self, entries: Collection[Tuple], quality_multiplier: float = 1):
        for e in entries:
            self.push_single(e)

    def _decode(self, enc):
        dec = cv2.imdecode(enc, 1)
        return self._to_tensor(dec).to(self._device)

    def get_all(self):
        data = list(self._memory.queue)
        imgs = [self._decode(x[0]) for x in data]
        labels = [x[1] for x in data]
        return list(zip(imgs, labels))

    @property
    def capacity(self):
        return self._capacity
