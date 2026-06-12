# in this file we will be importing twosecond audio samples and transforming our data for training our models


#import torchaudio
from data_import.google_speech_commands_dataset import load_twosec_google_speech_commands_for_training
from data_import.urban_sounds_dataset import load_twosec_urban_sounds_for_training
from data_import.esc50_dataset import load_twosec_esc50_for_training
from data_import.our_records_dataset import load_switch_off_records_for_training, load_turn_on_records_for_training, load_background_twosec_records_for_training

import torch
from torchaudio import transforms
import numpy as np
import random
import librosa


def twosec_data_import():
        
    google_dataset = np.asarray(load_twosec_google_speech_commands_for_training(), dtype = np.float32)
    urban_sounds_dataset = np.asarray(load_twosec_urban_sounds_for_training(), dtype = np.float32)
    esc50_dataset = np.asarray(load_twosec_esc50_for_training(), dtype = np.float32)
    turn_on_dataset = np.asarray(load_turn_on_records_for_training(), dtype = np.float32)
    switch_off_dataset = np.asarray(load_switch_off_records_for_training(), dtype = np.float32)
    back_ground_noises_dataset = load_background_twosec_records_for_training()


    cisza = np.asarray(back_ground_noises_dataset[2], dtype=np.float32)
    pokoj = np.asarray(back_ground_noises_dataset[1], dtype=np.float32)
    czytanie = np.asarray(back_ground_noises_dataset[0], dtype=np.float32)

# Do tego trzeba też wczytać pliki ciszy, pokoju i czytania, które trzeba jeszcze dograć

# będziemy wykonywać supervised learning, więc potrzebyjemy mieć etykiety do naszych danych treningowych

    google_dataset = [(sample, 0) for sample in google_dataset]  # etykieta 0 bo będą to background noice
    urban_sounds_dataset = [(sample, 0) for sample in urban_sounds_dataset]  # etykieta 0 bo będą to background noice
    esc50_dataset = [(sample, 0) for sample in esc50_dataset]  # etykieta 0 bo będą to background noice
    turn_on_dataset = [(sample, 1) for sample in turn_on_dataset]  # etykieta 1 bo będą to nagrania "turn on the leds"
    switch_off_dataset = [(sample, 2) for sample in switch_off_dataset]  # etykieta 2 bo będą to nagrania "switch off the leds"
    cisza = [(sample, 0) for sample in cisza]  # etykieta 0 bo będą to background noice
    pokoj = [(sample, 0) for sample in pokoj]  # etykieta 0 bo będą to background noice
    czytanie = [(sample, 0) for sample in czytanie] # etykieta 0 bo będą to background noice


    random.shuffle(google_dataset)
    google_dataset = google_dataset[:6930]
    google_dataset_training = google_dataset[:6300]
    google_dataset_validation = google_dataset[6300:]


    #print(f"Google dataset: {len(google_dataset)}, training (6300): {len(google_dataset_training)}, validation(630): {len(google_dataset_validation)}")


    random.shuffle(urban_sounds_dataset)
    urban_sounds_dataset = urban_sounds_dataset[:550]
    urban_sounds_dataset_training = urban_sounds_dataset[:500]
    urban_sounds_dataset_validation = urban_sounds_dataset[500:]


    #print(f"Urban Sounds dataset: {len(urban_sounds_dataset)}, training (500): {len(urban_sounds_dataset_training)}, validation(50): {len(urban_sounds_dataset_validation)}")


    random.shuffle(esc50_dataset)
    esc50_dataset = esc50_dataset[:550]
    esc50_dataset_training = esc50_dataset[:500]
    esc50_dataset_validation = esc50_dataset[500:]


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
    
    #print(f"Czytanie dataset: {len(czytanie)}, training (1000): {len(czytanie_training)}, validation(100): {len(czytanie_validation)}")


    random.shuffle(turn_on_dataset)
    turn_on_dataset_training = turn_on_dataset[:300]
    turn_on_dataset_validation = turn_on_dataset[300:]

    random.shuffle(switch_off_dataset)
    switch_off_dataset_training = switch_off_dataset[:300]
    switch_off_dataset_validation = switch_off_dataset[300:]
    
    #print(f"turn on dataset: {len(turn_on_dataset)}, training (300): {len(turn_on_dataset_training)}, validation(60): {len(turn_on_dataset_validating)}")
    #print(f"switch off dataset: {len(switch_off_dataset)}, training (300): {len(switch_off_dataset_training)}, validation(60): {len(switch_off_dataset_validating)}")

    
# Poniżej znajduje się fragment kodu odpowiedzialny za augmentację nagrań JARVIS_dataset_training 

    def augment_plus_audio(audio):
        return np.clip(audio * 1.1, -1.0, 1.0)


    def augment_minus_audio(audio):
        return np.clip(audio * 0.9, -1.0, 1.0)
    

    def augment_time_stretch_audio(audio, rate = 0.95):

        stretched = librosa.effects.time_stretch(
            audio,
            rate=rate
        )

        if len(stretched) > 16000:
            stretched = stretched[:16000]
        else:
            stretched = np.pad(
                stretched,
                (0, 16000 - len(stretched))
            )

        return stretched
        
    
    def augment_time_contracted_audio(audio, rate = 1.05):

        stretched = librosa.effects.time_stretch(
            audio,
            rate=rate
        )

        if len(stretched) > 16000:
            stretched = stretched[:16000]
        else:
            stretched = np.pad(
                stretched,
                (0, 16000 - len(stretched))
            )

        return stretched
    

    def augment_noise_audio(audio):

        noise = np.random.normal(0, 0.005, audio.shape)

        audio_augmented = np.clip(audio + noise, -1.0, 1.0)

        return audio_augmented



    turn_on_dataset_training = turn_on_dataset_training + [(augment_plus_audio(sample), label) for sample, label in turn_on_dataset_training] + [(augment_minus_audio(sample), label) for sample, label in turn_on_dataset_training] 
    turn_on_dataset_training = turn_on_dataset_training + [(augment_time_stretch_audio(sample), label) for sample, label in turn_on_dataset_training] + [(augment_time_contracted_audio(sample), label) for sample, label in turn_on_dataset_training]
    turn_on_dataset_training = turn_on_dataset_training + [(augment_noise_audio(sample), label) for sample, label in turn_on_dataset_training]
    
    random.shuffle(turn_on_dataset_training)

    #print(f"turn on dataset training after augmentation: {len(turn_on_dataset_training)}")

    #print(f"First augmented sample: {turn_on_dataset_training[0][0]}, Label: {turn_on_dataset_training[0][1]}")

    switch_off_dataset_training = turn_on_dataset_training + [(augment_plus_audio(sample), label) for sample, label in switch_off_dataset_training] + [(augment_minus_audio(sample), label) for sample, label in switch_off_dataset_training] 
    switch_off_dataset_training = turn_on_dataset_training + [(augment_time_stretch_audio(sample), label) for sample, label in switch_off_dataset_training] + [(augment_time_contracted_audio(sample), label) for sample, label in switch_off_dataset_training]
    switch_off_dataset_training = turn_on_dataset_training + [(augment_noise_audio(sample), label) for sample, label in switch_off_dataset_training]
    
    random.shuffle(switch_off_dataset_training)

    #print(f"switch off dataset training after augmentation: {len(switch_off_dataset_training)}")

    #print(f"First augmented sample: {switch_off_dataset_training[0][0]}, Label: {switch_off_dataset_training[0][1]}")



    raw_training_dataset = google_dataset_training + urban_sounds_dataset_training + esc50_dataset_training + switch_off_dataset_training + turn_on_dataset_training + cisza_training + pokoj_training + czytanie_training
    raw_validation_dataset = google_dataset_validation + urban_sounds_dataset_validation + esc50_dataset_validation + switch_off_dataset_validation + turn_on_dataset_validation + cisza_validation + pokoj_validation + czytanie_validation


    #print(f"Training dataset (11100): {len(raw_training_dataset)}, Validation dataset (1140): {len(raw_validation_dataset)}")


# Poniżej będziemy zamieniać dane na melspectrogramy:

    #Najpierw trzeba zrobić MelSpectrogramy z tych danych, bo model będzie trenowany na MelSpectrogramach, a nie na surowych danych audio. Poniżej definiuję transformację, która będzie zamieniać moje dane audio (numpy array) na MelSpectrogramy, które są potrzebne do trenowania modelu w PyTorch.
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
# 360 własnych nagrań (turn on the leds)
# 360 własnych nagrań (switch off the leds)
# 13230 nagrań z Google Speech Commands
# 1050 nagrań z Urban Sounds
# 1050 nagrań z ESC50
# 1100 nagrań ciszy
# 1650 nagrań pokoju jak się coś dzieje
# 1100 nagrań czytania
# 3850 nagrań z rozmów w salonie


# Struktura danych:

# Do trenowania modelu

# 300 nagrań własnych (turn on), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 36 = 10800)
# 300 nagrań własnych (switch off), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 36 = 10800)
# 12600 nagrań z Google Speech Commands
# 1000 nagrań z Urban Sounds
# 1000 nagrań z ESC50 
# 1000 nagrań ciszy
# 1500 nagrań pokoju jak się coś dzieje
# 1000 nagrań czytania
# 3500 nagrań z rozmów w salonie

# suma (turn_on+ switch off) = 21600
# suma (bez JARVIS) = 900 + 6300 + 500 + 500 + 1000 + 1600 = 21600
# razem 43200


# Do validacji modelu

# 60 własnych nagrań turn on zostawić do validacji na koniec
# 60 własnych nagrań swich off zostawić do validacji na koniec
# 630 nagrań z Google Speech Commands
# 50 nagrań z urban sounds
# 50 nagrań z ESC50
# 100 nagrań ciszy
# 150 nagrań pokoju jak się coś dzieje
# 100 nagrań czytania
# 350 nagrań z rozmów w salonie

# suma do validacji = 60 + 60 + 90 + 630 + 50 + 50 + 100 + 160 + 350 = 1550