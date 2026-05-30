# here we will be importing data for evaluation of our models

import matplotlib.pyplot as plt  
import librosa 
import os
import numpy as np  


# importing JARVIS records for evaluation (one second) ===================================================================================

def load_JARVIS_records_for_evaluation():
    dataset_JARVIS = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation_onesec_data/JARVIS"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 16000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (16000 próbek)
            JARVIS_dataset_for_evaluation.append(audio[:16000])  # jeśli tak, to bierzemy pierwsze 16000 próbek
            #print("Długość", len(turnon_dataset_for_evaluation[-1])/16000, "sekund")
            audio = audio[16000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 16000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            JARVIS_dataset_for_evaluation.append(audio_padded)
            #print("Długość", len(audio_padded)/16000, "sekund")

    return JARVIS_dataset_for_evaluation 

'''

data = load_JARVIS_records_for_evaluation()

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



# importing turn on the leds records for evaluation (two seconds) ===================================================================================

def load_turn_on_records_for_evaluation():
    dataset_turnon = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation_twosec_data/turn_on"

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

'''
data = load_turn_on_records_for_evaluation()

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



# importing switch off the leds records for evaluation (two seconds) ===================================================================================

def load_switch_off_records_for_evaluation():
    dataset_switchoff = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation_twosec_data/switch_off"

    for file in os.listdir(folder):

        full_path = os.path.join(folder, file)
        if file.endswith(".wav"):
            dataset_switchoff.append({
                "path": full_path
            })

    #print("Loaded:", len(dataset_switchoff))


    switchoff_dataset_for_evaluation = []

    for sample in dataset_switchoff:
        
        audio, sr = librosa.load(
            sample["path"],
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
        )

        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalizacja audio, dzielimy przez maksymalną wartość bezwzględną, aby mieć wartości w zakresie [-1, 1]

        if len(audio) >= 32000:   # tutaj sprawdzamy, czy długość audio jest większa lub równa 1 sekundzie (32000 próbek)
            switchoff_dataset_for_evaluation.append(audio[:32000])  # jeśli tak, to bierzemy pierwsze 32000 próbek
            #print("Długość", len(switchoff_dataset_for_evaluation[-1])/32000, "sekund")
            audio = audio[32000:]  # i usuwamy te próbki z oryginalnego audio, żeby sprawdzić resztę
        else:
            audio_padded = np.pad(audio, (0, 32000 - len(audio)), mode='constant')  # jeśli nie, to dopełniamy audio zerami do 1 sekundy
            switchoff_dataset_for_evaluation.append(audio_padded)
            #print("Długość", len(audio_padded)/32000, "sekund")

    return switchoff_dataset_for_evaluation 

'''
data = load_switch_off_records_for_evaluation()

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


# importing background noise records for evaluation (one second) ===================================================================================

def load_background_onesec_records_for_evaluation():
    dataset_background = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation_onesec_data/background_noise"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
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

    return background_dataset_for_evaluation 


'''
data = load_background_onesec_records_for_evaluation()

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


# importing background noise records for evaluation (two seconds) ===================================================================================

def load_background_twosec_records_for_evaluation():
    dataset_background = []

    folder = "/home/aleksander/studia/semestr4/AML/AML_Voice_recognizer/data/evaluation_twosec_data/background_noise"

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
            mono=True,   # tutaj ustawiamy mono, czyli pojedynczy kanał
            sr = 16000,   # tutaj ustawiamy docelową częstotliwość próbkowania na 16kHz
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


'''
data = load_background_twosec_records_for_evaluation()

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

