# this is the most important file. Here we have a loop which recognize commends


# below we will import tensors of voice and then run them through the model and then we will send the output to arduino_comm.py file which will communicate with arduino and turn on or off the leds
'''

    if prediction < 0.05: 
        line.set_color('b')
        ser.write(b'0')  # włącz LED
    else:
        line.set_color('r')
        ser.write(b'1')  # wyłącz LED

'''


# W tym pliku będę się uczuć tworzyć okno aplikacji w PySide oraz wyświetlać wykres w PySide za pomocą pyqtgraph

# https://www.pythonguis.com/tutorials/pyside6-creating-your-first-window/

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

from models.wake_word_model import *
from models.command_recognition_model import *
import pyqtgraph
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QFont, QIcon
from PySide6.QtSvg import QSvgRenderer
import numpy as np
import threading
import audio_stream
import serial
import torch
from torchaudio import transforms
import queue
import time

prediction_queue = queue.Queue()

commands_queue = queue.Queue()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model import:
model = CRNN_wake_word().to(device)
model.load_state_dict(torch.load("JARVIS_najlepszy.pth", map_location=device))
model.eval()

model_com = CRNN_commands().to(device)
model_com.load_state_dict(torch.load("command_new.pth", map_location=device))
model_com.eval()

#ser = serial.Serial('/dev/ttyACM0', 9600)  


mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


db_transform = transforms.AmplitudeToDB()


def JARVIS_model():
    while True:
        audio = audio_stream.audio_queue.get()

        model_input = audio / (np.max(np.abs(audio)) + 1e-8)  # nromalizacja do zakresu [-1, 1]
    
        model_input_torch = torch.tensor(model_input, dtype=torch.float32)

        mel = mel_transform(model_input_torch)

        mel = db_transform(mel)

        mel = mel.unsqueeze(0).unsqueeze(0)  # [1,1,64,T]

            
        with torch.no_grad(): # disables gradient tracking
            model_in = mel.to(device)
            
            logits = model(model_in)
            jarvis_prob = torch.softmax(logits, dim=1)[0, 1].item()   # procent szans na 1 chyba xd

            prediction_queue.put((audio, jarvis_prob))


            if jarvis_prob > 0.005: 
                commands_model()    # uruchom model komend 
            


def commands_model():
    audio = []
    i = 0
    while(i<40):
        temp = audio_stream.audio_queue.get()    
        audio.append(temp)
        i+=1
    
    model_input = [i / (np.max(np.abs(i)) + 1e-8) for i in audio]   # nromalizacja do zakresu [-1, 1]

    model_input_temp = []

    for i in range(0, len(model_input) - 10, 2):
        model_input_temp.append(np.concatenate((model_input[i], model_input[i + 10])))
        
    model_input_torch = [torch.tensor(i, dtype=torch.float32) for i in model_input_temp]

    mel = [mel_transform(i) for i in model_input_torch]

    mel = [db_transform(i) for i in mel]

    mel = [i.unsqueeze(0).unsqueeze(0) for i in mel]  # [1,1,64,T]

    predictions = []

    turn_on = 0
    switch_off = 0
    back = 0


    for mel in mel:
        with torch.no_grad(): # disables gradient tracking
            model_in = mel.to(device)
            
            output = model_com(model_in)
            prediction = torch.argmax(output, dim=1).item()
             
            if prediction == 2:
                switch_off += 1
            elif prediction == 1:
                turn_on += 1
            else: 
                back += 1

    print(f"back_ground = {back},    switch_off = {switch_off},    turn_on = {turn_on}")
'''
    if  turn_on > switch_off: 
        print("Włączamy ledy")
        ser.write(b'1')  # włącz LED
    elif turn_on < switch_off:
        print("Wyłączamy ledy")
        ser.write(b'0')  # wyłącz LED
''' 

    
JARVIS_model_thread = threading.Thread(target=JARVIS_model, daemon=True)
JARVIS_model_thread.start()

        
t = threading.Thread(target=audio_stream.start_stream, daemon=True)
t.start()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedSize(1200, 800)
        self.setWindowTitle("My App")
        # randomowy licznik do listening

        self.n = 0
        self.i = 0

        font = QFont("Segoe UI", 20)


        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.klikniecie_przycisku)
        self.button.setFixedSize(300, 100)
        self.button.setFont(font)



        self.status_label = QLabel("Program status: STOPPED")
        self.status_label.setFixedSize(500,100)
        
        # 3. Przypisanie czcionki do etykiety
        self.status_label.setFont(font)


        self.plot_widget = pyqtgraph.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showAxis('left', False)
        self.plot_widget.showAxis('bottom', False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.setXRange(0, 16000)
        self.plot_widget.setAntialiasing(True)
        
        self.curve = self.plot_widget.plot(pen=pyqtgraph.mkPen(color=(0, 200, 255), width=2))

        self.timer = QTimer()
        #self.timer.timeout.connect(self.update_plot)
        self.timer.timeout.connect(self.update_plot)


        


        # centralny widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # GŁÓWNY layout pionowy
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # === GÓRA: PLOT ===
        main_layout.addWidget(self.plot_widget, stretch=5)

        # === DÓŁ: HORIZONTAL ===
        bottom_layout = QHBoxLayout()

        # LEWA STRONA
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.button)

        # PRAWA STRONA
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.model_label = QLabel("JARVIS is NOT LISTENING")
        self.model_label.setFont(font)
        right_layout.addWidget(self.model_label)

        # dodanie do dolnego layoutu
        bottom_layout.addWidget(left_widget)
        bottom_layout.addWidget(right_widget)

        main_layout.addLayout(bottom_layout, stretch=1)



    def klikniecie_przycisku(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Program status: STOPPED")
            self.model_label.setText("JARVIS is NOT LISTENING")
            self.button.setText("Start")
            self.curve.setData([])
        else:
            self.timer.start(100)
            self.status_label.setText("Program status: ACTIVE")
            self.button.setText("Stop")


    def update_plot(self):
        temp = None

        while not prediction_queue.empty():
            temp = prediction_queue.get_nowait()

        if temp is None:
            return

        audio, pred = temp

        audio = np.array(audio)
        self.curve.setData(audio, skipFiniteCheck=True)

        if pred > 0.01:
            self.curve.setPen(pyqtgraph.mkPen(color=(255, 10, 10), width=2))
        else:
            self.curve.setPen(pyqtgraph.mkPen(color=(0, 200, 255), width=2))

        
        # animacja "LISTENING ..."
        if self.i % 3 == 0:
            self.n += 1
            dots = "." * (self.n % 4)

            self.model_label.setText(f"JARVIS is LISTENING {dots}")

        self.i += 1


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()



# Sens będzie taki, że jeśli przycisk zostanie wciśnięty, to wyświetli się nam grafika z pyqtgraph, na której będą się wyświetlać właśnie przykładowe randomowe audio co 0.1 s


