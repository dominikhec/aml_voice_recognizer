# here we weill load esc50 dataset, and here we will make little improvements over out data

import matplotlib.pyplot as plt            
import librosa 
import os
import numpy as np
from pathlib import Path


def load_onesec_esc50_for_training():
    dataset_esc50 = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "audio_ESC50"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_esc50.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_esc50))


    esc50_dataset_for_training = []

    for sample in dataset_esc50:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        while len(audio) >= 16000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            esc50_dataset_for_training.append(audio[:16000])  # jeśli tak, to bierzemy pierwsze 16000 próbek
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę


    return esc50_dataset_for_training  

def load_twosec_esc50_for_training():
    dataset_esc50 = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "audio_ESC50"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_esc50.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_esc50))


    esc50_dataset_for_training = []

    for sample in dataset_esc50:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        while len(audio) >= 32000:
            esc50_dataset_for_training.append(audio[:32000])
            audio = audio[32000:]


    return esc50_dataset_for_training  