from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from matplotlib.pyplot import imread

from utils.transform import ToTensor
"""
Loads the train/test set.
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.

Set data_root to point to the Train/Test folders.
"""


class notMNIST(Dataset):
    """Creating a sub class of torch.utils.data.dataset.Dataset
    """

    def __init__(self, data_root):
        """The init method is called when this class will be instantiated.
        """
        self.data = self._get_data(data_root)
        self.transform = transforms.Compose([ToTensor()])

    def _get_data(self, data_root):
        images_path, labels = [], []
        folders = listdir(data_root)

        for folder in folders:
            print(folder)
            folder_path = join(data_root, folder)
            for image_file in listdir(folder_path):
                try:
                    image_path = join(folder_path, image_file)
                    images_path.append(image_path)
                    labels.append(int(folder))  # Folders are A-J so labels will be 0-9
                    # labels.append(ord(folder)-65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, image_file))
        data = [(i, l) for i, l in zip(images_path, labels)]
        return data

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """The Dataloader is a generator that repeatedly calls the getitem method.

        getitem is supposed to return (X, labels, Z) for the specified index.
        """
        image_path, label = self.data[index]

        image = imread(image_path)
        image = image.reshape(28, 28)

        label = np.array(label)

        sample = {"image": image, "label": label, "image_path": image_path}

        if self.transform:
            sample = self.transform(sample)
        return sample
