# in this file we will have a command_recognition_model 


import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN_commands(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Dodanie BatchNorm dla stabilności
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(.3),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(.3)
        )

        self.gru = nn.GRU(256 * 4, 256, batch_first=True)
        self.fc = nn.Linear(256, 3)

        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(B, W, C * H)

        x, _ = self.gru(x)
        
        weights = torch.softmax(
            self.attention(x),
            dim=1
        )

        x = (weights * x).sum(dim=1)
        return self.fc(x)