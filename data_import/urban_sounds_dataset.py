# here we weill load urban sounds dataset, and here we will make little improvements over out data

import matplotlib.pyplot as plt            
import librosa 
import os
import numpy as np
from pathlib import Path

def load_onesec_urban_sounds_for_training():
    dataset_urban_sounds = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "urban_sounds"

    for label in os.listdir(folder):

        label_path = os.path.join(folder, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                full_path = os.path.join(label_path, file)

                dataset_urban_sounds.append({
                    "path": full_path
                })

    #print("Loaded:", len(dataset_urban_sounds))
    #print(dataset_urban_sounds[0])

    urban_sounds_dataset_for_training = []

    for sample in dataset_urban_sounds:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8) 
        
        while len(audio) >= 16000: 
            urban_sounds_dataset_for_training.append(audio[:16000])
            audio = audio[16000:]
            

    return urban_sounds_dataset_for_training  

def load_twosec_urban_sounds_for_training():
    dataset_urban_sounds = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "urban_sounds"

    for label in os.listdir(folder):

        label_path = os.path.join(folder, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                full_path = os.path.join(label_path, file)

                dataset_urban_sounds.append({
                    "path": full_path
                })

    #print("Loaded:", len(dataset_urban_sounds))
    #print(dataset_urban_sounds[0])

    urban_sounds_dataset_for_training = []

    for sample in dataset_urban_sounds:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True, 
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        

        while len(audio) >= 32000: 
            urban_sounds_dataset_for_training.append(audio[:32000]) 
            audio = audio[32000:]
            

    return urban_sounds_dataset_for_training  