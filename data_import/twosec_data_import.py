# in this file we will be importing twosecond audio samples and transforming our data for training our models

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

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


    cisza = np.asarray(back_ground_noises_dataset[5], dtype=np.float32)
    pokoj = np.asarray(back_ground_noises_dataset[4], dtype=np.float32)
    czytanie = np.asarray(back_ground_noises_dataset[0], dtype=np.float32)
    salon = np.asarray(back_ground_noises_dataset[1] + back_ground_noises_dataset[2] + back_ground_noises_dataset[3], dtype=np.float32)
    

    google_dataset = [(sample, 0) for sample in google_dataset]  # etykieta 0 bo będą to background noice
    urban_sounds_dataset = [(sample, 0) for sample in urban_sounds_dataset]  # etykieta 0 bo będą to background noice
    esc50_dataset = [(sample, 0) for sample in esc50_dataset]  # etykieta 0 bo będą to background noice
    turn_on_dataset = [(sample, 1) for sample in turn_on_dataset]  # etykieta 1 bo będą to nagrania "turn on the leds"
    switch_off_dataset = [(sample, 2) for sample in switch_off_dataset]  # etykieta 2 bo będą to nagrania "switch off the leds"
    cisza = [(sample, 0) for sample in cisza]  # etykieta 0 bo będą to background noice
    pokoj = [(sample, 0) for sample in pokoj]  # etykieta 0 bo będą to background noice
    czytanie = [(sample, 0) for sample in czytanie] # etykieta 0 bo będą to background noice
    salon = [(sample, 0) for sample in salon] # etykieta 0 bo będą to background noice


    random.shuffle(google_dataset)
    google_dataset = google_dataset[:13860]
    google_dataset_training = google_dataset[:12600]
    google_dataset_validation = google_dataset[12600:]

    #print(f"Google dataset: {len(google_dataset)}, training (12600): {len(google_dataset_training)}, validation(1260): {len(google_dataset_validation)}")


    random.shuffle(urban_sounds_dataset)
    urban_sounds_dataset = urban_sounds_dataset[:1100]
    urban_sounds_dataset_training = urban_sounds_dataset[:1000]
    urban_sounds_dataset_validation = urban_sounds_dataset[1000:]


    #print(f"Urban Sounds dataset: {len(urban_sounds_dataset)}, training (1000): {len(urban_sounds_dataset_training)}, validation(100): {len(urban_sounds_dataset_validation)}")


    random.shuffle(esc50_dataset)
    esc50_dataset = esc50_dataset[:1100]
    esc50_dataset_training = esc50_dataset[:1000]
    esc50_dataset_validation = esc50_dataset[1000:]


    #print(f"ESC50 dataset: {len(esc50_dataset)}, training (1000): {len(esc50_dataset_training)}, validation(100): {len(esc50_dataset_validation)}")

    
    random.shuffle(cisza)
    #print(f"Cisza dataset before shuffling: {len(cisza)}")
    cisza = cisza[:550]
    cisza_training = cisza[:500]
    cisza_validation = cisza[500:550]

    #print(f"Cisza dataset: {len(cisza)}, training (500): {len(cisza_training)}, validation(50): {len(cisza_validation)}")

    random.shuffle(pokoj)
    #print(f"Pokoj dataset before shuffling: {len(pokoj)}")
    pokoj = pokoj[:990]
    pokoj_training = pokoj[:900]
    pokoj_validation = pokoj[900:990]

    #print(f"Pokoj dataset: {len(pokoj)}, training (900): {len(pokoj_training)}, validation(90): {len(pokoj_validation)}")

    random.shuffle(czytanie)
    #print(f"Czytanie dataset before shuffling: {len(czytanie)}")
    czytanie = czytanie[:550]
    czytanie_training = czytanie[:500]
    czytanie_validation = czytanie[500:550]
    
    #print(f"Czytanie dataset: {len(czytanie)}, training (500): {len(czytanie_training)}, validation(50): {len(czytanie_validation)}")

    random.shuffle(salon)
    #print(f"Czytanie dataset before shuffling: {len(czytanie)}")
    salon = salon[:5610]
    salon_training = salon[:5100]
    salon_validation = salon[5100:5610]

    #print(f"Salon dataset: {len(salon)}, training (5100): {len(salon_training)}, validation(510): {len(salon_validation)}")

    random.shuffle(turn_on_dataset)
    turn_on_dataset_training = turn_on_dataset[:300]
    turn_on_dataset_validation = turn_on_dataset[300:]

    random.shuffle(switch_off_dataset)
    switch_off_dataset_training = switch_off_dataset[:300]
    switch_off_dataset_validation = switch_off_dataset[300:]

    #print(f"turn on dataset: {len(turn_on_dataset)}, training (300): {len(turn_on_dataset_training)}, validation(60): {len(turn_on_dataset_validating)}")
    #print(f"switch off dataset: {len(switch_off_dataset)}, training (300): {len(switch_off_dataset_training)}, validation(60): {len(switch_off_dataset_validating)}")

    
# Below we will augment our JARVIS recordings

    def augment_plus_audio(audio):
        return np.clip(audio * random.uniform(1.03, 1.1), -1.0, 1.0)


    def augment_minus_audio(audio):
        return np.clip(audio * random.uniform(0.9, 0.97), -1.0, 1.0)
    

    def augment_time_stretch_audio(audio, rate = None):

        if rate == None:              
                rate = random.uniform(0.92, 0.97)

        stretched = librosa.effects.time_stretch(audio, rate=rate)

        if len(stretched) > 32000:
            stretched = stretched[:32000]
        else:
            stretched = np.pad(stretched, (0, 32000 - len(stretched)))

        return stretched
        
    
    def augment_time_contracted_audio(audio, rate = None):
                                       
        if rate == None:
            rate = random.uniform(1.02, 1.07)

        stretched = librosa.effects.time_stretch(audio, rate=rate)

        if len(stretched) > 32000:
            stretched = stretched[:32000]
        else:
            stretched = np.pad(stretched, (0, 32000 - len(stretched)))

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

        if len(shifted) > 32000:
            shifted = shifted[:32000]
        else:
            shifted = np.pad(shifted, (0, 32000 - len(shifted)))

        return shifted



    turn_on_dataset_training = turn_on_dataset_training + [(augment_plus_audio(sample), label) for sample, label in turn_on_dataset_training] + [(augment_minus_audio(sample), label) for sample, label in turn_on_dataset_training] 
    turn_on_dataset_training = turn_on_dataset_training + [(augment_time_stretch_audio(sample), label) for sample, label in turn_on_dataset_training] + [(augment_time_contracted_audio(sample), label) for sample, label in turn_on_dataset_training]
    turn_on_dataset_training = turn_on_dataset_training + [(augment_noise_audio(sample), label) for sample, label in turn_on_dataset_training]
    turn_on_dataset_training = turn_on_dataset_training + [(augment_pitch_shift_audio(sample), label) for sample, label in turn_on_dataset_training]
    
    
    random.shuffle(turn_on_dataset_training)

    #print(f"turn on dataset training after augmentation: {len(turn_on_dataset_training)}")

    #print(f"First augmented sample: {turn_on_dataset_training[0][0]}, Label: {turn_on_dataset_training[0][1]}")

    switch_off_dataset_training = switch_off_dataset_training + [(augment_plus_audio(sample), label) for sample, label in switch_off_dataset_training] + [(augment_minus_audio(sample), label) for sample, label in switch_off_dataset_training] 
    switch_off_dataset_training = switch_off_dataset_training + [(augment_time_stretch_audio(sample), label) for sample, label in switch_off_dataset_training] + [(augment_time_contracted_audio(sample), label) for sample, label in switch_off_dataset_training]
    switch_off_dataset_training = switch_off_dataset_training + [(augment_noise_audio(sample), label) for sample, label in switch_off_dataset_training]
    switch_off_dataset_training = switch_off_dataset_training + [(augment_pitch_shift_audio(sample), label) for sample, label in switch_off_dataset_training]
    
    random.shuffle(switch_off_dataset_training)

    #print(f"switch off dataset training after augmentation: {len(switch_off_dataset_training)}")

    #print(f"First augmented sample: {switch_off_dataset_training[0][0]}, Label: {switch_off_dataset_training[0][1]}")



    raw_training_dataset = google_dataset_training + urban_sounds_dataset_training + esc50_dataset_training + switch_off_dataset_training + turn_on_dataset_training + cisza_training + pokoj_training + czytanie_training + salon_training
    raw_validation_dataset = google_dataset_validation + urban_sounds_dataset_validation + esc50_dataset_validation + switch_off_dataset_validation + turn_on_dataset_validation + cisza_validation + pokoj_validation + czytanie_validation + salon_validation


    #print(f"Training dataset (43200): {len(raw_training_dataset)}, Validation dataset (2280): {len(raw_validation_dataset)}")


# Below we will convert audio to mel spectograms

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


    random.shuffle(training_dataset)
    random.shuffle(validation_dataset)

    return training_dataset, validation_dataset
# To be downloaded:
# 360 own recordings (turn on the leds)
# 360 own recordings (switch off the leds)
# 13,860 recordings from Google Speech Commands
# 1,050 recordings from Urban Sounds
# 1,050 recordings from ESC50
# 550 recordings of silence
# 990 recordings of the room when something is happening
# 550 recordings of reading
# 5,610 recordings of conversations in the living room


# Data structure:

# For model training

# 300 own recordings (turn on), from which through augmentation we can make 5,400 recordings (300 * 36 = 10,800)
# 300 own recordings (switch off), from which through augmentation we can make 5,400 recordings (300 * 36 = 10,800)
# 12,600 recordings from Google Speech Commands
# 1,000 recordings from Urban Sounds
# 1,000 recordings from ESC50 
# 500 recordings of silence
# 900 recordings of the room when something is happening
# 500 recordings of reading
# 5,100 recordings of conversations in the living room

# sum (turn_on + switch off) = 21,600
# sum (without JARVIS) = 12,600 + 1,000 + 1,000 + 500 + 900 + 500 + 5,100 = 21,600
# total 43,200


# For model validation

# 60 own recordings turn on to be left for validation at the end
# 60 own recordings switch off to be left for validation at the end
# 1,260 recordings from Google Speech Commands
# 100 recordings from Urban Sounds
# 100 recordings from ESC50
# 50 recordings of silence
# 90 recordings of the room when something is happening
# 50 recordings of reading
# 510 recordings of conversations in the living room

# sum for validation = 60 + 60 + 1,260 + 100 + 100 + 50 + 90 + 50 + 510 = 2,28