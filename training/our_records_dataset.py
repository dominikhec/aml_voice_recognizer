# here we will import our own recorded files

import matplotlib.pyplot as plt  
import librosa 
import os
import numpy as np  



def load_JARVIS_records_for_training():
    dataset_JARVIS = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/training_and_validating/JARVIS"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 16000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            JARVIS_dataset_for_training.append(audio[:16000])  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(turnon_dataset_for_training[-1])/16000, "sekund")
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            JARVIS_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/16000, "sekund")

    return JARVIS_dataset_for_training 

'''
data = load_JARVIS_records_for_training()

print("Loaded:", len(data))

audio = data[1]
sampling_rate = 16000

print("Audio:", audio)
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(audio))
print("Total Duration:", len(audio) / sampling_rate, "seconds")


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''






def load_turn_on_records_for_training():
    dataset_turnon = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/training_and_validating/turn_on_leds"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            turnon_dataset_for_training.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(turnon_dataset_for_training[-1])/32000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            turnon_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return turnon_dataset_for_training 

'''
data = load_turn_on_records_for_training()

print("Loaded:", len(data))

audio = data[1]
sampling_rate = 16000

print("Audio:", audio)
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(audio))
print("Total Duration:", len(audio) / sampling_rate, "seconds")


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''








def load_switch_off_records_for_training():
    dataset_switchoff = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/training_and_validating/switch_off_leds"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            switchoff_dataset_for_training.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(switchoff_dataset_for_training[-1])/32000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            switchoff_dataset_for_training.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return switchoff_dataset_for_training 

'''
data = load_switch_off_records_for_training()

print("Loaded:", len(data))

audio = data[1]
sampling_rate = 16000

print("Audio:", audio)
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(audio))
print("Total Duration:", len(audio) / sampling_rate, "seconds")


plt.plot(audio)
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''





