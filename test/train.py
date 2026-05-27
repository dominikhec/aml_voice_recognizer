from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os

class AudioDataset(Dataset):

    def __init__(self, file_list, labels, sample_rate=16000):

        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate

        self.resampler = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        path = self.file_list[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path)

        # mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # resampling
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), torch.tensor(label)
    
train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root="./data",
    download=True,
    subset="training"
)

test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root="./data",
    download=True,
    subset="testing"
)
    
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

from CRNN import CRNN
model = CRNN(num_classes=10).to(device)

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()

for epoch in range(10):

    total_loss = 0

    for waveforms, labels in train_loader:

        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(waveforms)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss: {total_loss/len(train_loader):.4f}")