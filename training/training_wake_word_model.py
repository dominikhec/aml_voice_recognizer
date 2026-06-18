# in this file we will be training models on our data

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

from data_import.onesec_data_import import onesec_data_import
from models.wake_word_model import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



if __name__ == "__main__":
    
# Below is a piece of code in which we import data ready for training:

    data = onesec_data_import()

    training_set = data[0]
    validation_set = data[1]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = CRNN_wake_word().to(device)

    l_r = 0.001
    bs = 32
    # Below are two data loaders respectively one for train dataset and one for test dataset
    train_loader = DataLoader(training_set, batch_size = bs, shuffle = True)
    test_loader = DataLoader(validation_set, batch_size = bs, shuffle = False)


    optimizer = optim.Adam(model.parameters(), lr=l_r)
    criterion = nn.CrossEntropyLoss()


    print(f"Now testing for learning rate: {l_r} and batch size: {bs}")

    for epoch in range(5):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}") 

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy for learning rate: {l_r} and batch size: {bs}: {accuracy:.5f}%")

    print(f"Final loss for lr={l_r}, bs={bs}: {epoch_loss:.4f}")
    print(f"Final accuracy for lr={l_r}, bs={bs}: {accuracy:.5f}%")


    torch.save(model.state_dict(), "wake_word_model.pth")


