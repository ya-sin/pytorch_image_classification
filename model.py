import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset import notMNIST
from utils.net import Net

import re
import cv2

import matplotlib.pyplot as plt
# from parameters import MODEL_NAME, N_EPOCHS, BATCH_SIZE


class Classifier:
    def __init__(self,
                 dataset,
                 save_model_name,
                 batch_size
                 ):
        self.net = Net()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.save_model_name = save_model_name
        self.dataset_len = len(dataset)

    def train(self, epochs, learning_rate):

        print("-------- Training Step Start --------")

        loss_history = []
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        #epoch_loss = 0
        # n_batches = len(train_dataset) // self.batch_size

        for epoch in range(epochs):
            for step, data in enumerate(self.dataloader, 0):
                train_x, train_y = data["image"], data["label"]
                outputs = self.net.forward(train_x)

                # CrossEntropyLoss requires arg2 to be torch.LongTensor
                # print(train_x.size())
                # print(train_y.size())
                loss = self.criterion(outputs, train_y.long())
                #epoch_loss += loss.item()
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()
                optimizer.step()

                # There are len(dataset)/BATCH_SIZE batches.
                # We print the epoch loss when we reach the last batch.
                # if (step + 1) % 2000 == 1999:
                if step % 100 == 0:
                    #epoch_loss = epoch_loss / n_batches
                    loss_history.append(loss)
                    print("Epoch [{}/{}], Step {}, Loss {:.4f}".format(epoch + 1, epochs, step + 1, loss.item()))

        torch.save(self.net, 'models/{}.pt'.format(self.save_model_name))

        # plt.plot(np.array(range(1, epochs + 1)), loss_history)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # plt.show()

    def test(self, save_dir):
        print("loaded classifier")

        classifier = torch.load(self.save_model_name).eval()
        correct = 0
        saved = 0
        for _, data in enumerate(self.dataloader, 0):
            test_x, test_y, path = data["image"], data["label"], data["image_path"]
            pred = classifier.forward(test_x)
            y_hat = np.argmax(pred.data)

            print("pred: ", y_hat)
            print("truth: ", test_y)

            # read source image
            source_path = re.search("'([^']*)'", str(path)).group(1)
            img = cv2.imread(source_path, 0)

            # saved estimated image to right folder
            result = re.search('\(([^)]+)', str(y_hat)).group(1)
            p = os.path.sep.join([save_dir, result, "image_pred_{}.png".format(saved)])
            cv2.imwrite(p, img)

            # show source image
            if 0:
                a = np.transpose(test_x[0].numpy(), (1, 2, 0))
                b = a.squeeze()
                plt.figure()
                plt.imshow(b)
                plt.show()

            # calculate accuracy
            if y_hat == test_y:
                correct += 1
            saved += 1

        print("Accuracy={}".format(correct / self.dataset_len))
