import torch
import numpy as np

class ToTensor(object):
    def __call__(self, sample):
        image, label, image_path = sample["image"], sample["label"], sample["image_path"]
        image = image.reshape(1, 28, 28)

        return{
                "image":torch.from_numpy(image),
                "label":label,
                "image_path":image_path
                }
