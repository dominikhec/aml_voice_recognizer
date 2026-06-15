# this file will be collecting the audio from microphone and sending it to main.py file


import sounddevice as sd
import numpy as np
import queue


audio_queue = queue.Queue()

SAMPLE_RATE = 16000  # sample rate jaki chcemy uzyskiwać z mikrofonu
BLOCK_SIZE = 1600  # rejestrujemy jednorazowo 0.1 sekundy nagrania bo 16000 / 1600 = 0.1 s
BUFFER_SIZE = 16000  # rozmiar bufora do którego będziemy zbierać nagrania, żeby potem wysłać je do modelu (1 sekunda nagrania)


#BUFFER_SIZE_2 = 32000
#buffer_2 = np.zeros(BUFFER_SIZE_2) 

buffer = np.zeros(BUFFER_SIZE)  # buffer do którego będziemy zbierać nagrania, żeby potem wysłać je do modelu (1 sekunda nagrania), będziemy go przesuwać miejsce po miejscu co 0.1s 

def audio_callback(indata, frames, time, status):
    global buffer
    #global buffer_2

#indata - to są faktyczne dane które otrzymuje mikrofon, w formie tablicy numpy
#frames - liczba próbek w tym bloku (powinna być równa BLOCK_SIZE)  

    # chunk to nasz aktualny fragment dźwięku, który właśnie otrzymaliśmy z mikrofonu długość to 0.1 s
    chunk = indata[:, 0]  # zamieniamy od razu dźwięk na mono

    buffer = np.roll(buffer, -len(chunk)) # przesuwamy buffer w lewo
    #buffer_2 = np.roll(buffer_2, -len(chunk))

    buffer[-len(chunk):] = chunk   # opisujemy nowy chunk na koniec
    #buffer_2[-len(chunk):] = chunk

    #model_input = buffer.copy()     # teraz buffer ma zawsze 1 sekundę audio


    audio_queue.put((buffer.copy()))# wysyłamy ten 1 sekundowy fragment audio do main.py 



def start_stream():     # funkcja odczytu (podsłuchu) mikrofonu (uruchamiana w main)
    
    stream = sd.InputStream(                #tutaj ustawiamy parametry w których chcemy żeby był rejestrowany dźwięk z mikrofonu
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    
    with stream:        # tutaj zaczynamy podsłuch naszego mikrofonu
        while True:
            sd.sleep(1000)


