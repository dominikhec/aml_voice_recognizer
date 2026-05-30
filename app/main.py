# this is the most important file. Here we have a loop which recognize commends


# below we will import tensors of voice and then run them through the model and then we will send the output to arduino_comm.py file which will communicate with arduino and turn on or off the leds

import threading
import audio_stream
import matplotlib.pyplot as plt
import numpy as np

t = threading.Thread(target=audio_stream.start_stream)
t.start()


plt.ion()  # interactive mode

fig, ax = plt.subplots()

x = np.arange(16000)  # 1 sekunda
line, = ax.plot(x, np.zeros(16000))

ax.set_ylim(-1, 1)
ax.set_title("Live microphone signal")


while True:
    data = audio_stream.audio_queue.get()

    line.set_ydata(data)   # podmieniamy dane

    plt.pause(0.01)        # pozwala matplotlibowi odświeżyć GUI






