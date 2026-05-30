# this file will be collecting the audio from microphone and sending it to main.py file


import sounddevice as sd
import numpy as np
import queue

audio_queue = queue.Queue()

SAMPLE_RATE = 16000  # sample rate jaki chcemy uzyskiwać z mikrofonu
BLOCK_SIZE = 1600  # rejestrujemy jednorazowo 0.1 sekundy nagrania bo 16000 / 1600 = 0.1 s
BUFFER_SIZE = 16000  # rozmiar bufora do którego będziemy zbierać nagrania, żeby potem wysłać je do modelu (1 sekunda nagrania)

buffer = np.zeros(BUFFER_SIZE)

def audio_callback(indata, frames, time, status):
    global buffer

#indata - to są faktyczne dane które otrzymuje mikrofon, w formie tablicy numpy
#frames - liczba próbek w tym bloku (powinna być równa BLOCK_SIZE)    

    chunk = indata[:, 0]  # zamieniamy od razu dźwięk na mono

    # 🔁 przesuwamy buffer w lewo
    buffer = np.roll(buffer, -len(chunk))

    # ➕ dopisujemy nowy chunk na koniec
    buffer[-len(chunk):] = chunk

    # 🧠 teraz buffer ma zawsze 1 sekundę audio
    model_input = buffer.copy()

    # NORMALIZACJA (ważne dla ML)
    model_input = model_input / (np.max(np.abs(model_input)) + 1e-8)
    
    audio_queue.put(model_input)

    # 👉 TU WRZUCASZ DO MODELU
    # pred = model(torch.tensor(model_input))


def start_stream():     # tutaj zaczynamy podsłuch naszego mikrofonu
    
    stream = sd.InputStream(                #tutaj ustawiamy parametry w których chcemy żeby był rejestrowany dźwięk z mikrofonu
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    
    with stream:
        print("Listening...")
        while True:
            sd.sleep(1000)


