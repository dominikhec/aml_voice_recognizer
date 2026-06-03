# in this file we will have our wake_word_model

import torch
import torch.nn as nn
import torch.nn.functional as F


# Below I create a class with my model that I will be training and then testing
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2688, 64)  # Adjust the input size based on your actual input dimensions
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

