# here we weill load esc50 dataset, and here we will make little improvements over out data

import matplotlib.pyplot as plt            
import librosa 
import os


def load_esc50_for_training():
    dataset_esc50 = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/audio_ESC50"

    for file in os.listdir(folder):

        if file.endswith(".wav"):
            dataset_esc50.append({
                "path": file
            })

    #print("Loaded:", len(dataset_esc50))
    #print(dataset_esc50[0])



    esc50_dataset_for_training = []

    for sample in dataset_esc50:
        
        audio = librosa.load(
            sample["path"],
        )

        esc50_dataset_for_training.append(audio)

    # trzeba tutaj przekonwertować Sample rate na 16kHz
    # i długość pliku na 1 sekundę, czyli 16000 próbek
    # i dźwięk na mono, ale chyba już jest mono, więc to powinno być ok



    return esc50_dataset_for_training  

