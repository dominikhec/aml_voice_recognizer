# in this file we will be taking data from data folder and then we will be conwerting data, to be ready for training session
# its a file for importing google speech commands dataset
# and here we will make little improvements over out data


 
import matplotlib.pyplot as plt            
import librosa 
import os


def load_google_speech_commands_for_training():
    dataset_google = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/google_speech_commands"

    for label in os.listdir(folder):

        label_path = os.path.join(folder, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                full_path = os.path.join(label_path, file)

                dataset_google.append({
                    "label": label,
                    "path": full_path
                })

    #print("Loaded:", len(dataset_google))
    #print(dataset_google[0])


    '''
    sample = dataset_google[0]

    audio, sr = librosa.load(
        sample["path"],
        sr=16000,
        mono=True
    )

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    print(f"Label: {sample['label']}")

    plt.plot(audio)
    plt.show()
    '''


    google_dataset_for_training = []

    for sample in dataset_google:
        
        audio = librosa.load(
            sample["path"],
        )

        google_dataset_for_training.append(audio)


    return google_dataset_for_training  

