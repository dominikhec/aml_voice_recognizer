# in this file we will be training models on our data

import sys
import os
from pathlib import Path

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

bs = [16, 32, 64]
weight_decay = [1e-4, 5e-3, 1e-3]

import itertools

for bs, weight_decay in itertools.product(bs, weight_decay):


    if __name__ == "__main__":
    # Poniżej znajduje się kawałek kodu w którym importujemy gotowe do treningu dane:

        data = onesec_data_import()

        training_set = data[0]
        validation_set = data[1]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #model = SimpleCNN().to(device)  
        model = CRNN().to(device)

        l_r = 0.001
        bs = 32
        # Below I create two data loader respectively one for train dataset and one for test dataset
        train_loader = DataLoader(training_set, batch_size = bs, shuffle = True)
        test_loader = DataLoader(validation_set, batch_size = bs, shuffle = False)


        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )


        print(f"Now testing for learning rate: {l_r} and batch size: {bs}")

        epochs = 5

        for epoch in range(5):
            model.train()   #ustawiamy nasz model na tryb treningowy (jest to istotne dla warstw takich jak Dropout czy BatchNorm)
            running_loss = 0.0  #zmienna pomocnicza służąca do sumowania strat w każdej epoce

            for data, target in train_loader:   #pętla po batchach danych treningowych
                # data to batch obraów, a target to batch etykiet/labelów (cyfr 0-9)
                data, target = data.to(device), target.to(device)   # przenosimy dane i labele na urządzenie obliczeniowe (CPU)

                # poniżej zerujemy gradienty zawarte w opimizerze przed rozpoczęciem obliczania nowych dla następnego batcha
                optimizer.zero_grad() # PyTorch accumulates gradients by default, so we need to clear them before computing new ones
                
                output = model(data)    # model przepuszcza dane przez sieć (output to tensor z przewidywaniami dla batcha)
                loss = criterion(output, target)    # liczymy błąd (loss) dla tego batcha
                loss.backward()    # backpropagacja, czyli pytorch liczy gradienty wag w siecie na podstawie loss

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()    # aktualizuje wagi modelu na nowe, lekko zoptymalizowane
                
                running_loss += loss.item()   # loss.item() = wartość liczby zmiennoprzecinkowej z tensora, sumujemy loss dla batchy, żeby później policzyć średni loss dla epoki
            
            epoch_loss = running_loss / len(train_loader)   # oblcizamy średni loss na epokę

            # poniżej wypisujemy dla odpowiednich epok, jaki otrzymaliśmy średni loss
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")     # można powiedzieć, że ta linijka wypisuje postęp treningu w konsoli

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
            print(f"Test Accuracy for learning rate: {l_r} and batch size: {bs}: {accuracy:.5f}%")

            scheduler.step(epoch_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(f"LR: {current_lr:.6f}")



        print(f"Final loss for lr={l_r}, bs={bs}: {epoch_loss:.4f}")
        print(f"Final accuracy for lr={l_r}, bs={bs}: {accuracy:.5f}%")


        current_dir = Path(__file__).resolve().parent
    
        path = current_dir.parent / f"wake_word_model_bs_{bs}_wd_{weight_decay}.pth"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #model = SimpleCNN().to(device)  
    model = CRNN().to(device)

    l_r = 1e-3
    bs = bs
    # Below I create two data loader respectively one for train dataset and one for test dataset
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
        patience=3
    )


    print(f"Now testing for learning rate: {l_r} and batch size: {bs}")

    epochs = 5

    for epoch in range(5):
        model.train()   #ustawiamy nasz model na tryb treningowy (jest to istotne dla warstw takich jak Dropout czy BatchNorm)
        running_loss = 0.0  #zmienna pomocnicza służąca do sumowania strat w każdej epoce

        for data, target in train_loader:   #pętla po batchach danych treningowych
            # data to batch obraów, a target to batch etykiet/labelów (cyfr 0-9)
            data, target = data.to(device), target.to(device)   # przenosimy dane i labele na urządzenie obliczeniowe (CPU)

            # poniżej zerujemy gradienty zawarte w opimizerze przed rozpoczęciem obliczania nowych dla następnego batcha
            optimizer.zero_grad() # PyTorch accumulates gradients by default, so we need to clear them before computing new ones
            
            output = model(data)    # model przepuszcza dane przez sieć (output to tensor z przewidywaniami dla batcha)
            loss = criterion(output, target)    # liczymy błąd (loss) dla tego batcha
            loss.backward()    # backpropagacja, czyli pytorch liczy gradienty wag w siecie na podstawie loss

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()    # aktualizuje wagi modelu na nowe, lekko zoptymalizowane
            
            running_loss += loss.item()   # loss.item() = wartość liczby zmiennoprzecinkowej z tensora, sumujemy loss dla batchy, żeby później policzyć średni loss dla epoki
        
        epoch_loss = running_loss / len(train_loader)   # oblcizamy średni loss na epokę

        # poniżej wypisujemy dla odpowiednich epok, jaki otrzymaliśmy średni loss
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")     # można powiedzieć, że ta linijka wypisuje postęp treningu w konsoli

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
        print(f"Test Accuracy for weight decay: {weight_decay} and batch size: {bs}: {accuracy:.5f}%")

        scheduler.step(accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"LR: {current_lr:.6f}")



    print(f"Final loss for lr={l_r}, bs={bs}: {epoch_loss:.4f}")
    print(f"Final accuracy for lr={l_r}, bs={bs}: {accuracy:.5f}%")

    torch.save(model.state_dict(), f"wake_word_model_wd_{str(weight_decay).replace('.',',')}_bs_{bs}.pth")

