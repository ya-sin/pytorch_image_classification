import torch
from torch.utils.data import DataLoader
from utils.dataset import notMNIST
import os
import json
import numpy as np

from model import Classifier
import matplotlib.pyplot as plt

CONFIG_PATH = "./config.json"

if __name__ == "__main__":

    with open(CONFIG_PATH) as cb:
        config = json.loads(cb.read())

        data = notMNIST(
            config["test"]["test_data_folder"]
        )

        model = Classifier(
            data,
            config["test"]["classifier"],
            config["test"]["batch_size"]
        )

        model.test(
            config["test"]["save_dir"]
        )
