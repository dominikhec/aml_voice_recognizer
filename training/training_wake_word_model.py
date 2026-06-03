# in this file we will be training models on our data


#import torchaudio
from google_speech_commands_dataset import load_onesec_google_speech_commands_for_training
from urban_sounds_dataset import load_onesec_urban_sounds_for_training
from esc50_dataset import load_onesec_esc50_for_training
from our_records_dataset import load_JARVIS_records_for_training, load_background_onesec_records_for_training

import torch
from torchaudio import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


if __name__ == "__main__":
    
    google_dataset = load_onesec_google_speech_commands_for_training()
    urban_sounds_dataset = load_onesec_urban_sounds_for_training()
    esc50_dataset = load_onesec_esc50_for_training()
    JARVIS_dataset = load_JARVIS_records_for_training()
    back_ground_noises_dataset = load_background_onesec_records_for_training()


    cisza = back_ground_noises_dataset[2]
    pokoj = back_ground_noises_dataset[1]
    czytanie = back_ground_noises_dataset[0]


    google_dataset = [(sample, 0) for sample in google_dataset]  # etykieta 0 bo będą to background noice
    urban_sounds_dataset = [(sample, 0) for sample in urban_sounds_dataset]  # etykieta 0 bo będą to background noice
    esc50_dataset = [(sample, 0) for sample in esc50_dataset]  # etykieta 0 bo będą to background noice
    JARVIS_dataset = [(sample, 1) for sample in JARVIS_dataset]  # etykieta 1 bo będą to nagrania "JARVIS"
    cisza = [(sample, 0) for sample in cisza]  # etykieta 0 bo będą to background noice
    pokoj = [(sample, 0) for sample in pokoj]  # etykieta 0 bo będą to background noice
    czytanie = [(sample, 0) for sample in czytanie] # etykieta 0 bo będą to background noice
    

    random.shuffle(google_dataset)
    google_dataset = google_dataset[:6930]
    google_dataset_training = google_dataset[:6300]
    google_dataset_validation = google_dataset[6300:]

    print(f"Google dataset: {len(google_dataset)}, training (6300): {len(google_dataset_training)}, validation(630): {len(google_dataset_validation)}")
    
    random.shuffle(urban_sounds_dataset)
    urban_sounds_dataset = urban_sounds_dataset[:550]
    urban_sounds_dataset_training = urban_sounds_dataset[:500]
    urban_sounds_dataset_validation = urban_sounds_dataset[500:]

    print(f"Urban Sounds dataset: {len(urban_sounds_dataset)}, training (500): {len(urban_sounds_dataset_training)}, validation(50): {len(urban_sounds_dataset_validation)}")
    
    random.shuffle(esc50_dataset)
    esc50_dataset = esc50_dataset[:550]
    esc50_dataset_training = esc50_dataset[:500]
    esc50_dataset_validation = esc50_dataset[500:]

    print(f"ESC50 dataset: {len(esc50_dataset)}, training (500): {len(esc50_dataset_training)}, validation(50): {len(esc50_dataset_validation)}")

    random.shuffle(cisza)
    print(f"Cisza dataset before shuffling: {len(cisza)}")
    cisza = cisza[:1100]
    cisza_training = cisza[:1000]
    cisza_validation = cisza[1000:1100]

    print(f"Cisza dataset: {len(cisza)}, training (1000): {len(cisza_training)}, validation(100): {len(cisza_validation)}")

    random.shuffle(pokoj)
    print(f"Pokoj dataset before shuffling: {len(pokoj)}")
    pokoj = pokoj[:1650]
    pokoj_training = pokoj[:1500]
    pokoj_validation = pokoj[1500:1650]

    print(f"Pokoj dataset: {len(pokoj)}, training (1500): {len(pokoj_training)}, validation(150): {len(pokoj_validation)}")

    random.shuffle(czytanie)
    print(f"Czytanie dataset before shuffling: {len(czytanie)}")
    czytanie = czytanie[:1100]
    czytanie_training = czytanie[:1000]
    czytanie_validation = czytanie[1000:1100]

    print(f"Czytanie dataset: {len(czytanie)}, training (1000): {len(czytanie_training)}, validation(100): {len(czytanie_validation)}")

    random.shuffle(JARVIS_dataset)
    JARVIS_dataset_training = JARVIS_dataset[:300]
    JARVIS_dataset_validation = JARVIS_dataset[300:]

    print(f"JARVIS dataset: {len(JARVIS_dataset)}, training (300): {len(JARVIS_dataset_training)}, validation(60): {len(JARVIS_dataset_validation)}")


    training_dataset = google_dataset_training + urban_sounds_dataset_training + esc50_dataset_training + JARVIS_dataset_training + cisza_training + pokoj_training + czytanie_training
    validation_dataset = google_dataset_validation + urban_sounds_dataset_validation + esc50_dataset_validation + JARVIS_dataset_validation + cisza_validation + pokoj_validation + czytanie_validation


    print(f"Training dataset (11100): {len(training_dataset)}, Validation dataset (1140): {len(validation_dataset)}")



    # Powinienem najpierw pobrać dane, popodpisywać, potasować i wtedy dopiero rozdzielić na trening i validację. 
    # Następnie należy zrobić augmentację nagrań JARVIS_training i wtedy trzeba przetasować jeszcze raz
    # następnie zamienić na mel spectogramy


    # Należy pobrać:
    # 360 własnych nagrań (JARVIS)
    # 6930 nagrań z Google Speech Commands
    # 550 nagrań z Urban Sounds
    # 550 nagrań z ESC50
    # 1100 nagrań ciszy
    # 1650 nagrań pokoju jak się coś dzieje
    # 1100 nagrań czytania


    # Struktura danych:

    # Do trenowania modelu

    # 300 nagrań własnych (JARVIS), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 18 = 5400)
    # 6300 nagrań z Google Speech Commands
    # 500 nagrań z Urban Sounds
    # 500 nagrań z ESC50 
    # 1000 nagrań ciszy
    # 1500 nagrań pokoju jak się coś dzieje
    # 1000 nagrań czytania

    # suma (JARVIS) = 5400
    # suma (bez JARVIS) = 900 + 6300 + 500 + 500 + 1000 + 1600 = 10800


    # Do validacji modelu

    # 60 własnych nagrań zostawić do validacji na koniec   
    # 630 nagrań z Google Speech Commands
    # 50 nagrań z urban sounds
    # 50 nagrań z ESC50
    # 100 nagrań ciszy
    # 150 nagrań pokoju jak się coś dzieje
    # 100 nagrań czytania

    # suma do validacji = 60 + 90 + 630 + 50 + 50 + 100 + 160 = 1140



