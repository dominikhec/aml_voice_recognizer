# in this file we will be taking data from data folder and then we will be conwerting data, to be ready for training session

# and here we will make little improvements over out data


import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
import IPython 
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import os
import pysofaconventions as sofa


'''
# The dataset is automatically downloaded if not available and 10 of each word is selected
dataset_google = pra.datasets.GoogleSpeechCommands(download=True, subset=10, seed=0)

# print dataset info, first 10 entries, and all sounds
print(dataset_google)
dataset_google.head(n=10)
print("All sounds in the dataset:")
print(dataset_google.classes)


folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/sofa"

sofa_files = [f for f in os.listdir(folder) if f.endswith(".sofa")]

sofas = []

for f in sofa_files:
    full_path = os.path.join(folder, f)
    
    sofa_file = sofa.SOFAFile(full_path, 'r')
    sofas.append(sofa_file)

print(f"Loaded {len(sofas)} SOFA files")

print(sofas)

'''


audio, sampling_rate = torchaudio.load("/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/test_python.wav")

audio = audio.mean(dim=0)   # ze względu na to, że audio jest stereo, a my chcemy mono, bierzemy średnią z obu kanałów

print("Audio:", audio)
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(audio))
print("Total Duration:", len(audio) / sampling_rate, "seconds")

IPython.display.Audio(audio, rate=16000)


plt.plot(audio)
plt.show()