# in this file we will have our wake_word_model

import torch
import torch.nn as nn
import torch.nn.functional as F


# Below  is a lass with model for training 
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2688, 64) 
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    



class CRNN_wake_word(nn.Module):
    def __init__(self, out_channels=2):

        super().__init__()
        
        # Dodanie BatchNorm dla stabilności
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(.25)
        )

        self.gru1 = nn.GRU(256 * 8, 256, batch_first=True)
        self.gru2 = nn.GRU(256, 512, batch_first=True)
        self.fc = nn.Linear(512, 2)

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(B, W, C * H)
        
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        
        weights = torch.softmax(
            self.attention(x),
            dim=1
        )

        x = (weights * x).sum(dim=1)
        return self.fc(x)


