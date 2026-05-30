# in this file we will be taking data from data folder and then we will be conwerting data, to be ready for training session
# its a file for importing google speech commands dataset
# and here we will make little improvements over out data


 
import matplotlib.pyplot as plt            
import librosa 
import numpy as np
import os
import math 


def load_onesec_google_speech_commands_for_training():
    dataset_google = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/google_speech_commands"

    for label in os.listdir(folder):

        label_path = os.path.join(folder, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                full_path = os.path.join(label_path, file)

                dataset_google.append({
                    "label": label,
                    "path": full_path
                })


    google_dataset_for_training = []

    for sample in dataset_google:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr=16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio

        google_dataset_for_training.append(audio)


    return google_dataset_for_training  


'''
data = load_onesec_google_speech_commands_for_training()

print("Loaded:", len(data))

audio = data[0]


print("Audio:", audio)
print("Total Samples in Audio:", len(audio))


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''

def load_twosec_google_speech_commands_for_training():
    dataset_google = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/google_speech_commands"

    for label in os.listdir(folder):

        label_path = os.path.join(folder, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                full_path = os.path.join(label_path, file)

                dataset_google.append({
                    "label": label,
                    "path": full_path
                })


    google_dataset_for_training = []


    print(len(dataset_google))

    for sample in dataset_google:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,
            sr=16000,
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio

        google_dataset_for_training.append(audio)
    

    google_dataset_for_training_1 = []
        
    for i in range(0, len(google_dataset_for_training)-1, 2):

        merged_audio = np.concatenate(
            (
                google_dataset_for_training[i],
                google_dataset_for_training[i+1]
            )
        )

        google_dataset_for_training_1.append(merged_audio)

    return google_dataset_for_training_1  


'''
data = load_twosec_google_speech_commands_for_training()

print("Loaded:", len(data))

audio = data[0]


print("Audio:", audio)
print("Total Samples in Audio:", len(audio))


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()

'''

