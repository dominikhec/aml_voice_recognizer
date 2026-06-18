# in this file we will be importing one second audio samples and transforming our data for training our models

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

#import torchaudio
from data_import.google_speech_commands_dataset import load_onesec_google_speech_commands_for_training
from data_import.urban_sounds_dataset import load_onesec_urban_sounds_for_training
from data_import.esc50_dataset import load_onesec_esc50_for_training
from data_import.our_records_dataset import load_JARVIS_records_for_training, load_background_onesec_records_for_training

import torch
from torchaudio import transforms
import numpy as np
import random
import librosa



def onesec_data_import():

# Poniżej znajduje się kawałek kodu odpowiadający za pobieranie danych z plików, podpisywanie labelów, tasowanie oraz dzielenia na zbiór treningowy i walidacyjny.

    google_dataset = np.asarray(load_onesec_google_speech_commands_for_training(), dtype=np.float32)
    urban_sounds_dataset = np.asarray(load_onesec_urban_sounds_for_training(), dtype=np.float32)
    esc50_dataset = np.asarray(load_onesec_esc50_for_training(), dtype=np.float32)
    JARVIS_dataset = np.asarray(load_JARVIS_records_for_training(), dtype=np.float32)
    back_ground_noises_dataset = load_background_onesec_records_for_training()


    cisza = np.asarray(back_ground_noises_dataset[0], dtype=np.float32)
    salon = np.asarray(back_ground_noises_dataset[1], dtype= np.float32)
    pokoj = np.asarray(back_ground_noises_dataset[4], dtype=np.float32)
    czytanie = np.asarray(back_ground_noises_dataset[5], dtype=np.float32)


    google_dataset = [(sample, 0) for sample in google_dataset]  # etykieta 0 bo będą to background noice
    urban_sounds_dataset = [(sample, 0) for sample in urban_sounds_dataset]  # etykieta 0 bo będą to background noice
    esc50_dataset = [(sample, 0) for sample in esc50_dataset]  # etykieta 0 bo będą to background noice
    JARVIS_dataset = [(sample, 1) for sample in JARVIS_dataset]  # etykieta 1 bo będą to nagrania "JARVIS"
    cisza = [(sample, 0) for sample in cisza]  # etykieta 0 bo będą to background noice
    pokoj = [(sample, 0) for sample in pokoj]  # etykieta 0 bo będą to background noice
    czytanie = [(sample, 0) for sample in czytanie] # etykieta 0 bo będą to background noice
    salon = [(sample, 0) for sample in salon] # etykieta 0 bo będą to background noice
    

    random.shuffle(google_dataset)
    google_dataset = google_dataset[:13230]
    google_dataset_training = google_dataset[:12600]
    google_dataset_validation = google_dataset[12600:]

    #print(f"Google dataset: {len(google_dataset)}, training (6300): {len(google_dataset_training)}, validation(630): {len(google_dataset_validation)}")
    
    random.shuffle(urban_sounds_dataset)
    urban_sounds_dataset = urban_sounds_dataset[:1050]
    urban_sounds_dataset_training = urban_sounds_dataset[:1000]
    urban_sounds_dataset_validation = urban_sounds_dataset[1000:]

    #print(f"Urban Sounds dataset: {len(urban_sounds_dataset)}, training (500): {len(urban_sounds_dataset_training)}, validation(50): {len(urban_sounds_dataset_validation)}")
    
    random.shuffle(esc50_dataset)
    esc50_dataset = esc50_dataset[:1050]
    esc50_dataset_training = esc50_dataset[:1000]
    esc50_dataset_validation = esc50_dataset[1000:]

    #print(f"ESC50 dataset: {len(esc50_dataset)}, training (500): {len(esc50_dataset_training)}, validation(50): {len(esc50_dataset_validation)}")

    random.shuffle(cisza)
    #print(f"Cisza dataset before shuffling: {len(cisza)}")
    cisza = cisza[:1100]
    cisza_training = cisza[:1000]
    cisza_validation = cisza[1000:1100]

    #print(f"Cisza dataset: {len(cisza)}, training (1000): {len(cisza_training)}, validation(100): {len(cisza_validation)}")

    random.shuffle(pokoj)
    #print(f"Pokoj dataset before shuffling: {len(pokoj)}")
    pokoj = pokoj[:1650]
    pokoj_training = pokoj[:1500]
    pokoj_validation = pokoj[1500:1650]

    #print(f"Pokoj dataset: {len(pokoj)}, training (1500): {len(pokoj_training)}, validation(150): {len(pokoj_validation)}")

    random.shuffle(czytanie)
    #print(f"Czytanie dataset before shuffling: {len(czytanie)}")
    czytanie = czytanie[:1100]
    czytanie_training = czytanie[:1000]
    czytanie_validation = czytanie[1000:1100]

    random.shuffle(czytanie)
    #print(f"Salon dataset before shuffling: {len(salon)}")
    salon = salon[:3850]
    salon_training = salon[:3500]
    salon_validation = salon[3500:3850]

    #print(f"Czytanie dataset: {len(czytanie)}, training (1000): {len(czytanie_training)}, validation(100): {len(czytanie_validation)}")

    random.shuffle(JARVIS_dataset)
    JARVIS_dataset_training = JARVIS_dataset[:300]
    JARVIS_dataset_validation = JARVIS_dataset[300:]

    #print(f"JARVIS dataset: {len(JARVIS_dataset)}, training (300): {len(JARVIS_dataset_training)}, validation(60): {len(JARVIS_dataset_validation)}")

    
# Poniżej znajduje się fragment kodu odpowiedzialny za augmentację nagrań JARVIS_dataset_training 

    def augment_plus_audio(audio):
        return np.clip(audio * random.uniform(1.03, 1.1), -1.0, 1.0)


    def augment_minus_audio(audio):
        return np.clip(audio * random.uniform(0.9, 0.97), -1.0, 1.0)
    

    def augment_time_stretch_audio(audio, rate = None):

        if rate == None:              
                rate = random.uniform(0.92, 0.97)

        stretched = librosa.effects.time_stretch(audio, rate=rate)

        if len(stretched) > 16000:
            stretched = stretched[:16000]
        else:
            stretched = np.pad(stretched, (0, 16000 - len(stretched)))

        return stretched
        
    
    def augment_time_contracted_audio(audio, rate = None):
                                       
        if rate == None:
            rate = random.uniform(1.02, 1.07)

        stretched = librosa.effects.time_stretch(audio, rate=rate)

        if len(stretched) > 16000:
            stretched = stretched[:16000]
        else:
            stretched = np.pad(stretched, (0, 16000 - len(stretched)))

        return stretched


    def augment_noise_audio(audio):

        noise = np.random.normal(0, random.uniform(0.02, 0.07), audio.shape)

        audio_augmented = np.clip(audio + noise, -1.0, 1.0)

        return audio_augmented
    

    def augment_pitch_shift_audio(audio, sr=16000, n_steps=None):
        # Zmienia wysokość dźwięku o losową liczbę półtonów.
        
        if n_steps is None:
            n_steps = random.uniform(-2.0, 2.0)

        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

        if len(shifted) > 16000:
            shifted = shifted[:16000]
        else:
            shifted = np.pad(shifted, (0, 16000 - len(shifted)))

        return shifted



    JARVIS_dataset_training = JARVIS_dataset_training + [(augment_plus_audio(sample), label) for sample, label in JARVIS_dataset_training] + [(augment_minus_audio(sample), label) for sample, label in JARVIS_dataset_training] 
    JARVIS_dataset_training = JARVIS_dataset_training + [(augment_time_stretch_audio(sample), label) for sample, label in JARVIS_dataset_training] + [(augment_time_contracted_audio(sample), label) for sample, label in JARVIS_dataset_training]
    JARVIS_dataset_training = JARVIS_dataset_training + [(augment_noise_audio(sample), label) for sample, label in JARVIS_dataset_training]
    JARVIS_dataset_training = JARVIS_dataset_training + [(augment_pitch_shift_audio(sample), label) for sample, label in JARVIS_dataset_training]
    
    random.shuffle(JARVIS_dataset_training)

    #print(f"JARVIS dataset training after augmentation: {len(JARVIS_dataset_training)}")

    #print(f"First augmented sample: {JARVIS_dataset_training[0][0]}, Label: {JARVIS_dataset_training[0][1]}")

    raw_training_dataset = google_dataset_training + urban_sounds_dataset_training + esc50_dataset_training + JARVIS_dataset_training + cisza_training + pokoj_training + czytanie_training + salon_training
    raw_validation_dataset = google_dataset_validation + urban_sounds_dataset_validation + esc50_dataset_validation + JARVIS_dataset_validation + cisza_validation + pokoj_validation + czytanie_validation + salon_validation

    #print(f"Training dataset (32400): {len(raw_training_dataset)}, Validation dataset (1490): {len(raw_validation_dataset)}")


# Poniżej będziemy zamieniać dane na melspectrogramy:

    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    db_transform = transforms.AmplitudeToDB()

    training_dataset = []

    for audio, label in raw_training_dataset:

        audio = torch.tensor(audio, dtype=torch.float32)

        mel = mel_transform(audio)

        mel = db_transform(mel)

        mel = mel.unsqueeze(0)

        training_dataset.append((mel, label))


    validation_dataset = []

    for audio, label in raw_validation_dataset:

        audio = torch.tensor(audio, dtype=torch.float32)

        mel = mel_transform(audio)

        mel = db_transform(mel)

        mel = mel.unsqueeze(0)

        validation_dataset.append((mel, label))


    # tasowanie danych treningowych
    random.shuffle(training_dataset)
    random.shuffle(validation_dataset)

    return training_dataset, validation_dataset




# Należy pobrać:
# 360 własnych nagrań (JARVIS)
# 13230 nagrań z Google Speech Commands
# 1050 nagrań z Urban Sounds
# 1050 nagrań z ESC50
# 1100 nagrań ciszy
# 1650 nagrań pokoju jak się coś dzieje
# 1100 nagrań czytania
# 3850 nagrań z rozmów w salonie


# Struktura danych:

# Do trenowania modelu

# 300 nagrań własnych (JARVIS), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 36 = 10800)
# 12600 nagrań z Google Speech Commands
# 1000 nagrań z Urban Sounds
# 1000 nagrań z ESC50 
# 1000 nagrań ciszy
# 1500 nagrań pokoju jak się coś dzieje
# 1000 nagrań czytania
# 3500 nagrań z rozmów w salonie

# suma (JARVIS) = 10800
# suma (bez JARVIS) = 900 + 6300 + 500 + 500 + 1000 + 1600 = 21600


# Do validacji modelu

# 60 własnych nagrań zostawić do validacji na koniec   
# 630 nagrań z Google Speech Commands
# 50 nagrań z urban sounds
# 50 nagrań z ESC50
# 100 nagrań ciszy
# 150 nagrań pokoju jak się coś dzieje
# 100 nagrań czytania
# 350 nagrań z rozmów w salonie

# suma do validacji = 60 + 90 + 630 + 50 + 50 + 100 + 160 + 350 = 1490