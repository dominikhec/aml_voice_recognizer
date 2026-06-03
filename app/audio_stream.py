# this file will be collecting the audio from microphone and sending it to main.py file

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

from models.wake_word_model import SimpleCNN
import sounddevice as sd
import numpy as np
import queue
import torch
from torchaudio import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model import:
model = SimpleCNN()
model.load_state_dict(torch.load("wake_word_model.pth", map_location=device))
model = SimpleCNN().to(device)
model.eval()


mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

db_transform = transforms.AmplitudeToDB()


audio_queue = queue.Queue()

SAMPLE_RATE = 16000  # sample rate jaki chcemy uzyskiwać z mikrofonu
BLOCK_SIZE = 1600  # rejestrujemy jednorazowo 0.1 sekundy nagrania bo 16000 / 1600 = 0.1 s
BUFFER_SIZE = 16000  # rozmiar bufora do którego będziemy zbierać nagrania, żeby potem wysłać je do modelu (1 sekunda nagrania)

buffer = np.zeros(BUFFER_SIZE)  # buffer do którego będziemy zbierać nagrania, żeby potem wysłać je do modelu (1 sekunda nagrania), będziemy go przesuwać miejsce po miejscu co 0.1s 

def audio_callback(indata, frames, time, status):
    global buffer

#indata - to są faktyczne dane które otrzymuje mikrofon, w formie tablicy numpy
#frames - liczba próbek w tym bloku (powinna być równa BLOCK_SIZE)  

    # chunk to nasz aktualny fragment dźwięku, który właśnie otrzymaliśmy z mikrofonu długość to 0.1 s
    chunk = indata[:, 0]  # zamieniamy od razu dźwięk na mono

    buffer = np.roll(buffer, -len(chunk)) # przesuwamy buffer w lewo

    buffer[-len(chunk):] = chunk   # opisujemy nowy chunk na koniec

    model_input = buffer.copy()     # teraz buffer ma zawsze 1 sekundę audio

    model_input = model_input / (np.max(np.abs(model_input)) + 1e-8)  # nromalizacja do zakresu [-1, 1]
    
    # teraz ten model_input powinniśmy wrzucić do modelu żeby sprawdzić co mówimy
    # pred = model(torch.tensor(model_input))


    model_input = torch.tensor(model_input, dtype=torch.float32)

    mel = mel_transform(model_input)

    mel = db_transform(mel)

    mel = mel.unsqueeze(0).unsqueeze(0)  # [1,1,64,T]

        
    with torch.no_grad(): # disables gradient tracking
            model_in = mel.to(device)
            
            logits = model(model_in)
            jarvis_prob = torch.softmax(logits, dim=1)[0, 1].item()

            #print(f"Jarvis probability: {jarvis_prob:.3f}")

            '''
            if jarvis_prob > 0.6:
                 prediction = 1
            else:
                 prediction = 0
            '''

    audio_queue.put((mel.cpu().numpy(), jarvis_prob))# wysyłamy ten 1 sekundowy fragment audio do main.py 

    # powinniśmy też wysłać wynik modelu do arduino_comm.py żeby włączyć lub wyłączyć led



def start_stream():     # funkcja odczytu (podsłuchu) mikrofonu (uruchamiana w main)
    
    stream = sd.InputStream(                #tutaj ustawiamy parametry w których chcemy żeby był rejestrowany dźwięk z mikrofonu
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    
    with stream:        # tutaj zaczynamy podsłuch naszego mikrofonu
        print("Listening...")
        while True:
            sd.sleep(1000)


