# this is the most important file. Here we have a loop which recognize commends


# below we will import tensors of voice and then run them through the model and then we will send the output to arduino_comm.py file which will communicate with arduino and turn on or off the leds

import threading
import audio_stream
import matplotlib.pyplot as plt
import numpy as np

t = threading.Thread(target=audio_stream.start_stream)
t.start()


plt.ion()  # interactive mode dla plotu

fig, ax = plt.subplots()    # tworzymy figure i axis dla naszego plotu

x = np.arange(16000)        # zmienna x reprezentująca 1 sekundę
line, = ax.plot(x, np.zeros(16000))     # inicjalizujemy linię z zerami

ax.set_ylim(-1, 1)      # ograniczenia na osi y (normalizacja do [-1, 1])
ax.set_title("Live microphone signal")  # tytuł wykresu


while True:     # główna pętla programu, która będzie czekać na nowe dane z mikrofonu i aktualizować wykres
    data = audio_stream.audio_queue.get()       # czekamy aż pojawią się nowe dane z mikrofonu (1 sekunda audio)

    line.set_ydata(data)   # podmieniamy stare dane na nowe    

    plt.pause(0.01)        # pozwala matplotlibowi odświeżyć GUI






