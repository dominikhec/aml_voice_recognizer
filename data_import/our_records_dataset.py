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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        training_audio_slot = []

        while(len(audio) >= 16000):   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            audio_1 = audio[:16000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            training_audio_slot.append(audio_1)  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(background_dataset_for_training[-1])/16000, "sekund")
            audio = audio[8000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę

        audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        training_audio_slot.append(audio_padded)
        #print("Długość", len(audio_padded)/16000, "sekund")
        background_dataset_for_training.append(training_audio_slot)


    return background_dataset_for_training 


'''
data = load_background_onesec_records_for_training()

print("loaded 1( to powinno być najdłuższe bo to są nagrania z salonu z rozmów):", len(data[0]))
print("Loaded 1:", len(data[1]))
print("Loaded 2:", len(data[2]))
print("Loaded 3:", len(data[3]))    #to jest cisza

slot_1 = data[0]
sampling_rate = 16000

print("Audio:", slot_1[15])
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(slot_1[15]))
print("Total Duration:", len(slot_1[15]) / sampling_rate, "seconds")


plt.plot(slot_1[15])
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
'''


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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        training_audio_slot = []

        while(len(audio) >= 32000):   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            audio_1 = audio[:32000]
            audio_1 = audio_1 / (np.max(np.abs(audio_1)) + 1e-8)
            training_audio_slot.append(audio_1)  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(background_dataset_for_training[-1])/16000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę

        audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
        audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)
        training_audio_slot.append(audio_padded)
        #print("Długość", len(audio_padded)/32000, "sekund")
        background_dataset_for_training.append(training_audio_slot)


    return background_dataset_for_training 


'''
data = load_background_twosec_records_for_training()

print("Loaded 1:", len(data[0]))    # to jest czytanie
print("Loaded 2:", len(data[1]))    # to jest pokój
print("Loaded 3:", len(data[2]))    # to jest cisza


slot_1 = data[0]
sampling_rate = 16000

print("Audio:", slot_1[15])
print("Sampling Rate:", sampling_rate)
print("Total Samples in Audio:", len(slot_1[15]))
print("Total Duration:", len(slot_1[15]) / sampling_rate, "seconds")


plt.plot(slot_1[15])
plt.grid(True)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()

'''




