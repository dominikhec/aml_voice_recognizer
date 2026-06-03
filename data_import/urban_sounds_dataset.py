# here we weill load urban sounds dataset, and here we will make little improvements over out data

import matplotlib.pyplot as plt            
import librosa 
import os
import numpy as np


def load_onesec_urban_sounds_for_training():
    dataset_urban_sounds = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/training_and_validating/urban_sounds"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstot
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]
        

        while len(audio) >= 16000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            urban_sounds_dataset_for_training.append(audio[:16000])  # jeśli tak, to bierzemy pierwsze 16000 próbek
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
            

    return urban_sounds_dataset_for_training  

'''
data = load_onesec_urban_sounds_for_training()

print("Loaded:", len(data))

audio= data[0]


print("Audio:", audio)
print("Total Samples in Audio:", len(audio))


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''

def load_twosec_urban_sounds_for_training():
    dataset_urban_sounds = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/training_and_validating/urban_sounds"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstot
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]
        

        while len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 2 sekundzie (32000 próbek)
            urban_sounds_dataset_for_training.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
            

    return urban_sounds_dataset_for_training  

'''
data = load_twosec_urban_sounds_for_training()

print("Loaded:", len(data))

audio= data[0]


print("Audio:", audio)
print("Total Samples in Audio:", len(audio))


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()

'''