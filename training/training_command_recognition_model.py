# in this file we will be training command_recognition_model on our data

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

#import torchaudio
from data_import.twosec_data_import import twosec_data_import
from models.wake_word_model import SimpleCNN


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



if __name__ == "__main__":
    
# Poniżej znajduje się kawałek kodu w którym importujemy gotowe do treningu dane:

    data = twosec_data_import()

    training_set = data[0]
    validation_set = data[1]

    print("lenght of training set: ",len(training_set))
    print("lenght of validation set: ", len(validation_set))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = SimpleCNN().to(device)  

# co tutuaj trzeba zrobić:

# 1. zmnieszyć ilość danych treningowych background noise do powiedzmy 5 000 - 10 000.
# 2. Zrobić więcej nagrań "turn on the leds" i "switch off the leds" poprzez augmentację audio (np. dodając szum, zmieniając pitch, zmniejszanie/zwiększanie szybkośći o 5%, zmiana głośności o 5% itp.)
# 3. Weighted Cross Entropy - Obliczać wagi odwrotnie proporcjonalne do liczności klas.
# 4. Ja bym proponował zrobić na przykład 5 modelu o trochę innych parametrach, zrobić tablicę z różnymi learning rate i batch size i potem wybrać najlepszy model na podstawie wyników na zbiorze walidacyjnym. Można też zrobić ensemble tych modeli, czyli np. średnia ważona ich predykcji, żeby uzyskać lepsze wyniki.
# 5. Trzeba napisać cały proces epochy trenowania z tymi lossami i optymalizatorami, żeby model się trenował.
# 6. Na koniec trzeba zapisać wagi modelu i wysłać go do evaluacji w innym folderze o nazwie "evaluation" gdzie porównamy wszystkie modele, learning raty i batch size.






