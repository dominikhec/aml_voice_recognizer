# here we weill load urban sounds dataset, and here we will make little improvements over out data

import matplotlib.pyplot as plt            
import librosa 
import os


def load_urban_sounds_for_training():
    dataset_urban_sounds = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/urban_sounds"

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
        
        audio = librosa.load(
            sample["path"],
        )

        urban_sounds_dataset_for_training.append(audio)

    # trzeba tutaj przekonwertować Sample rate na 16kHz
    # i długość pliku na 1 sekundę, czyli 16000 próbek
    # i dźwięk na mono, ale chyba już jest mono, więc to powinno być ok

    
    return urban_sounds_dataset_for_training  
    

