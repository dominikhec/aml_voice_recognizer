# in this file we will be training models on our data

import sys
import os
from pathlib import Path
import time



project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

from data_import.twosec_data_import import twosec_data_import
from models.command_recognition_model import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

bs = [32]
weight_decay = [1e-4]

import itertools

for bs, weight_decay in itertools.product(bs, weight_decay):


    if __name__ == "__main__":

        data = twosec_data_import()

        training_set = data[0]
        validation_set = data[1]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        model = CRNN_commands().to(device)

        l_r = 0.001
        bs = bs
        # Below  is two data loaders respectively one for train dataset and one for test dataset
        train_loader = DataLoader(training_set, batch_size = bs, shuffle = True)
        test_loader = DataLoader(validation_set, batch_size = bs, shuffle = False)


        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=1,
        )


        print(f"Now testing for weight_decay: {weight_decay} and batch size: {bs}")

        start = time.time()

        epochs = 5

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for data, target in train_loader:
                
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)

            end = time.time()

            
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad(): # disables gradient tracking
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.5f}%")

            scheduler.step(epoch_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(f"LR: {current_lr:.6f}")
            print(f"Trainig time: {(end - start)/60} min")

        print(f"Final loss for lr={l_r}, bs={bs}: {epoch_loss:.4f}")
        print(f"Final accuracy for lr={l_r}, bs={bs}: {accuracy:.5f}%")
        

        torch.save(model.state_dict(), f"command_word_model_wd_{str(weight_decay).replace('.',',')}_bs_{bs}.pth")
