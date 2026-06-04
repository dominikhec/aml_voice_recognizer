# in this file we will be importing twosecond audio samples and transforming our data for training our models


#import torchaudio
from google_speech_commands_dataset import load_twosec_google_speech_commands_for_training
from urban_sounds_dataset import load_twosec_urban_sounds_for_training
from esc50_dataset import load_twosec_esc50_for_training
from our_records_dataset import load_switch_off_records_for_training, load_turn_on_records_for_training

import torch
from torchaudio import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


if __name__ == "__main__":


    def twosec_data_import():
        
        google_dataset = np.asarray(load_twosec_google_speech_commands_for_training(), dtype = np.float32)
        urban_sounds_dataset = load_twosec_urban_sounds_for_training()
        esc50_dataset = load_twosec_esc50_for_training()
        turn_on_dataset = load_turn_on_records_for_training()
        switch_off_dataset = load_switch_off_records_for_training()

    # będziemy wykonywać supervised learning, więc potrzebyjemy mieć etykiety do naszych danych treningowych

        google_dataset = [(sample, 0) for sample in google_dataset]  # etykieta 0 bo będą to background noice
        urban_sounds_dataset = [(sample, 0) for sample in urban_sounds_dataset]  # etykieta 0 bo będą to background noice
        esc50_dataset = [(sample, 0) for sample in esc50_dataset]  # etykieta 0 bo będą to background noice
        turn_on_dataset = [(sample, 1) for sample in turn_on_dataset]  # etykieta 1 bo będą to nagrania "turn on the leds"
        switch_off_dataset = [(sample, 2) for sample in switch_off_dataset]  # etykieta 2 bo będą to nagrania "switch off the leds"


        raw_dataset = google_dataset + urban_sounds_dataset + esc50_dataset + turn_on_dataset + switch_off_dataset


        #Najpierw trzeba zrobić MelSpectrogramy z tych danych, bo model będzie trenowany na MelSpectrogramach, a nie na surowych danych audio. Poniżej definiuję transformację, która będzie zamieniać moje dane audio (numpy array) na MelSpectrogramy, które są potrzebne do trenowania modelu w PyTorch.
        mel_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        db_transform = transforms.AmplitudeToDB()

        train_dataset = []

        for audio, label in raw_dataset:

            audio = torch.tensor(audio, dtype=torch.float32)

            mel = mel_transform(audio)

            mel = db_transform(mel)

            mel = mel.unsqueeze(0)

            train_dataset.append((mel, label))


        # tasowanie danych treningowych
        random.shuffle(train_dataset)

        return train_dataset




# Należy pobrać:
# 360 własnych nagrań (turn on the leds)
# 360 własnych nagrań (switch off the leds)
# 6930 nagrań z Google Speech Commands
# 550 nagrań z Urban Sounds
# 550 nagrań z ESC50
# 1100 nagrań ciszy
# 1650 nagrań pokoju jak się coś dzieje
# 1100 nagrań czytania


# Struktura danych:

# Do trenowania modelu

# 300 nagrań własnych (turn on), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 18 = 5400)
# 300 nagrań własnych (switch off), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 18 = 5400)
# 6300 nagrań z Google Speech Commands
# 500 nagrań z Urban Sounds
# 500 nagrań z ESC50 
# 1000 nagrań ciszy
# 1500 nagrań pokoju jak się coś dzieje
# 1000 nagrań czytania

# suma (turn on + switch off) = 10800
# suma (bez JARVIS) = 900 + 6300 + 500 + 500 + 1000 + 1600 = 10800


# Do validacji modelu

# 60 własnych nagrań turn on zostawić do validacji na koniec
# 60 własnych nagrań switch off zostawić do validacji na koniec   
# 630 nagrań z Google Speech Commands
# 50 nagrań z urban sounds
# 50 nagrań z ESC50
# 100 nagrań ciszy
# 150 nagrań pokoju jak się coś dzieje
# 100 nagrań czytania

# suma do validacji = 60 + 60 + 90 + 630 + 50 + 50 + 100 + 160 = 1200
