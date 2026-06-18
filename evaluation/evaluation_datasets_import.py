# here we will be importing data for evaluation of our models

import matplotlib.pyplot as plt  
import librosa 
import os
import numpy as np  
import random
from torchaudio import transforms
import torch


# importing JARVIS records for evaluation (one second)

def load_JARVIS_records_for_evaluation():
    dataset_JARVIS = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation/evaluation_onesec_data/JARVIS"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_JARVIS.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_JARVIS))


    JARVIS_dataset_for_evaluation = []

    for sample in dataset_JARVIS:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True, 
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        if len(audio) >= 16000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            JARVIS_dataset_for_evaluation.append(audio[:16000])  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(turnon_dataset_for_evaluation[-1])/16000, "sekund")
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            JARVIS_dataset_for_evaluation.append(audio_padded)
            #print("Długość", len(audio_padded)/16000, "sekund")

    JARVIS_dataset_for_evaluation = np.asarray(JARVIS_dataset_for_evaluation, dtype=np.float32)

    JARVIS_dataset_for_evaluation = [(sample, 1) for sample in JARVIS_dataset_for_evaluation]

    random.shuffle(JARVIS_dataset_for_evaluation)

    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    db_transform = transforms.AmplitudeToDB()

    evaluation_dataset = []

    for audio, label in JARVIS_dataset_for_evaluation:

        audio = torch.tensor(audio, dtype=torch.float32)

        mel = mel_transform(audio)

        mel = db_transform(mel)

        mel = mel.unsqueeze(0)

        evaluation_dataset.append((mel, label))


    return evaluation_dataset 


# importing turn on the leds records for evaluation (two seconds) ===================================================================================

def load_turn_on_records_for_evaluation():
    dataset_turnon = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation/evaluation_twosec_data/turn_on"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_turnon.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_turnon))


    turnon_dataset_for_evaluation = []

    for sample in dataset_turnon:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            turnon_dataset_for_evaluation.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(turnon_dataset_for_evaluation[-1])/32000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            turnon_dataset_for_evaluation.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")


    return turnon_dataset_for_evaluation 

# importing switch off the leds records for evaluation (two seconds) ===================================================================================

def load_switch_off_records_for_evaluation():
    dataset_switchoff = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation/evaluation_twosec_data/switch_off"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_switchoff.append({
                "path": full_path
            })

    switchoff_dataset_for_evaluation = []

    for sample in dataset_switchoff:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        if len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            switchoff_dataset_for_evaluation.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(switchoff_dataset_for_evaluation[-1])/32000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            switchoff_dataset_for_evaluation.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return switchoff_dataset_for_evaluation 

# importing background noise records for evaluation (one second) ===================================================================================

def load_background_onesec_records_for_evaluation():
    dataset_background = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation/evaluation_onesec_data/background_noise"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_background.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_background))


    background_dataset_for_evaluation = []

    for sample in dataset_background:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True, 
            sr = 16000,
        )


        while(len(audio) >= 16000):   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            audio_1 = audio[:16000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            background_dataset_for_evaluation.append(audio_1)  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(background_dataset_for_evaluation[-1])/16000, "sekund")
            audio = audio[8000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę

        audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        background_dataset_for_evaluation.append(audio_padded)
        #print("Długość", len(audio_padded)/16000, "sekund")

    background_dataset_for_evaluation = np.asarray(background_dataset_for_evaluation, dtype=np.float32)

    background_dataset_for_evaluation = [(sample, 0) for sample in background_dataset_for_evaluation]

    random.shuffle(background_dataset_for_evaluation)

    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    db_transform = transforms.AmplitudeToDB()

    evaluation_dataset = []

    for audio, label in background_dataset_for_evaluation:

        audio = torch.tensor(audio, dtype=torch.float32)

        mel = mel_transform(audio)

        mel = db_transform(mel)

        mel = mel.unsqueeze(0)

        evaluation_dataset.append((mel, label))


    return evaluation_dataset 

# importing background noise records for evaluation (two seconds) ===================================================================================

def load_background_twosec_records_for_evaluation():
    dataset_background = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation/evaluation_twosec_data/background_noise"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_background.append({
                "path": full_path
            })

    background_dataset_for_evaluation = []

    for sample in dataset_background:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        while(len(audio) >= 32000):   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            audio_1 = audio[:32000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            background_dataset_for_evaluation.append(audio_1)  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(background_dataset_for_evaluation[-1])/32000, "sekund")
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę

        audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        background_dataset_for_evaluation.append(audio_padded)
        #print("Długość", len(audio_padded)/32000, "sekund")

    return background_dataset_for_evaluation 
