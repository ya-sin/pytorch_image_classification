import json

from utils.dataset import notMNIST
from model import Classifier

from torch.utils.data import DataLoader

CONFIG_PATH = "./config.json"

if __name__ == "__main__":

    with open(CONFIG_PATH) as cb:
        config = json.loads(cb.read())

        data = notMNIST(
            config["train"]["train_data_folder"]
        )

        model = Classifier(
            data,
            config["train"]["save_model_name"],
            config["train"]["batch_size"]
        )

        model.train(
            config["train"]["epochs"],
            config["train"]["learning_rate"]
        )

        #test_data = notMNIST(
         #   config["test"]["test_data_folder"]
        #)

        #model.test(
          #  test_data,
         #   config["test"]["classifier"]
        #)
