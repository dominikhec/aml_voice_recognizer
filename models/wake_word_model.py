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
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        # Obliczamy wagę ważności dla każdego kroku czasowego
        weights = torch.softmax(self.attn(x), dim=1) 
        # Ważona suma: [batch, hidden_size]
        context = torch.sum(x * weights, dim=1)      
        return context 

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
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
            nn.Dropout2d(0.25)
        )

        self.lstm1 = nn.LSTM(..., 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)

        self.attention = Attention(512)

        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(B, W, C * H)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.attention(x)
        
        return self.fc(x)


