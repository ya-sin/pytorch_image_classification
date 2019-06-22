import torch.nn as nn
import torch
import torch.nn.functional as F


# All torch models have to inherit from the Module class
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self._fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.5)
        )
        self._fc2 = nn.Sequential(
            nn.Linear(512, 9)
        )

    def forward(self, x):

        x = self._hidden1(x)
        x = self._hidden2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self._fc1(x)
        x = self._fc2(x)

        # Reshaping the tensor to BATCH_SIZE x 320. Torch infers this from other dimensions when one of the parameter is -1.
        return x
# import torch.nn as nn
# import torch
# import torch.nn.functional as F


# # All torch models have to inherit from the Module class
# class Model(torch.nn.Module):

#   def __init__(self):
#       super(Model, self).__init__()
#       # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#       # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

#       self.conv3 = nn.Conv2d(1, 10, kernel_size=3,stride=1,padding=1)
#       self.conv4 = nn.Conv2d(10, 20, kernel_size=3,stride=1,padding=1)


#       self.conv4_drop = nn.Dropout2d()
#       self.fc1 = nn.Linear(980, 490)
#       self.fc2 = nn.Linear(490, 9)

#   def forward(self, x):
#       x = F.relu(F.max_pool2d(self.conv3(x), 2))
#       x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))


#       # x = F.relu(F.max_pool2d(self.conv1(x), 2))
#       # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

#       # Reshaping the tensor to BATCH_SIZE x 320. Torch infers this from other dimensions when one of the parameter is -1.
#       x = x.view(-1, 980)
#       x = F.relu(self.fc1(x))
#       x = F.dropout(x)
#       x = self.fc2(x)
#       return x
