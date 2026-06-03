# in this file we will be training models on our data


#import torchaudio
from data_import.onesec_data_import import onesec_data_import

import torch
from torchaudio import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import librosa


if __name__ == "__main__":
    
# Poniżej znajduje się kawałek kodu w którym importujemy gotowe do treningu dane:

    data = onesec_data_import()

    training_set = data[0]
    validation_set = data[1]

    print(f"Length of training set: {len(training_set)}")
    print(f"Length of validation set: {len(validation_set)}")



