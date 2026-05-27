import torch
import torch.nn as nn
import torchaudio

class CRNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64
        )

        # CNN
        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # RNN
        self.lstm = nn.LSTM(
            input_size=64 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        # waveform -> spectrogram
        x = self.melspec(x)

        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)

        # x:
        # [batch, channels, freq, time]

        b, c, f, t = x.shape

        # reshape for LSTM
        x = x.permute(0, 3, 1, 2)

        x = x.reshape(b, t, c * f)

        # LSTM
        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x
    
class MFCC_CNN(nn.Module):

    def __init__(self, num_classes=10, sample_rate=16000):

        super().__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 64
            }
        )

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        # x: [batch, samples]

        x = self.mfcc(x)          # [batch, mfcc, time]

        x = x.unsqueeze(1)        # [batch, 1, mfcc, time]

        x = self.cnn(x)

        x = x.squeeze(-1).squeeze(-1)

        x = self.fc(x)

        return x