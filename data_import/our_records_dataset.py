# here we will import our own recorded files

import matplotlib.pyplot as plt  
import librosa 
import os
import numpy as np  
from pathlib import Path


def load_JARVIS_records_for_training():
    dataset_JARVIS = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "JARVIS"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_JARVIS.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_JARVIS))


    JARVIS_dataset_for_training = []

    for sample in dataset_JARVIS:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        if len(audio) >= 16000:
            JARVIS_dataset_for_training.append(audio[:16000])
            audio = audio[16000:]
        else:
            audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant') 
            JARVIS_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/16000, "sekund")

    return JARVIS_dataset_for_training 

def load_turn_on_records_for_training():
    dataset_turnon = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "turn_on_leds"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_turnon.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_turnon))


    turnon_dataset_for_training = []

    for sample in dataset_turnon:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True, 
            sr = 16000,   
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        if len(audio) >= 32000:
            turnon_dataset_for_training.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(turnon_dataset_for_training[-1])/32000, "sekund")
            audio = audio[32000:]
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')
            turnon_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return turnon_dataset_for_training 

def load_switch_off_records_for_training():
    dataset_switchoff = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "switch_off_leds"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_switchoff.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_switchoff))


    switchoff_dataset_for_training = []

    for sample in dataset_switchoff:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        if len(audio) >= 32000:
            switchoff_dataset_for_training.append(audio[:32000])
            #print("Długość", len(switchoff_dataset_for_training[-1])/32000, "sekund")
            audio = audio[32000:]
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')
            switchoff_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return switchoff_dataset_for_training 


# import ciszy, nagrań z pokoju i czytania  [cisza, pokój, czytanie, rozmowy w salonie]

def load_background_onesec_records_for_training():
    dataset_background = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "background_noise"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_background.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_background))


    background_dataset_for_training = []

    for sample in dataset_background:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000,
        )

        training_audio_slot = []

        while(len(audio) >= 16000):
            audio_1 = audio[:16000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            training_audio_slot.append(audio_1)
            #print("Długość", len(background_dataset_for_training[-1])/16000, "sekund")
            audio = audio[8000:]

        audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        training_audio_slot.append(audio_padded)
        #print("Długość", len(audio_padded)/16000, "sekund")
        background_dataset_for_training.append(training_audio_slot)


    return background_dataset_for_training 

# import ciszy, nagrań z pokoju i czytania  [cisza, pokój, czytanie]

def load_background_twosec_records_for_training():
    dataset_background = []

    current_dir = Path(__file__).resolve().parent
    
    folder = current_dir.parent / "data" / "training_and_validating" / "background_noise"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_background.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_background))


    background_dataset_for_training = []

    for sample in dataset_background:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr = 16000, 
        )

        training_audio_slot = []

        while(len(audio) >= 32000):
            audio_1 = audio[:32000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            training_audio_slot.append(audio_1)
            #print("Długość", len(background_dataset_for_training[-1])/16000, "sekund")
            audio = audio[16000:]

        audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        training_audio_slot.append(audio_padded)
        #print("Długość", len(audio_padded)/32000, "sekund")
        background_dataset_for_training.append(training_audio_slot)


    return background_dataset_for_training 
