# this is the most important file. Here we have a loop which recognize commends

# W tym pliku będę się uczuć tworzyć okno aplikacji w PySide oraz wyświetlać wykres w PySide za pomocą pyqtgraph
    
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
from PySide6.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush
import numpy as np
import threading
import audio_stream
import serial
import torch
from torchaudio import transforms
import queue

prediction_queue = queue.Queue()

commands_prediction_queue = queue.Queue()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model import:
model = CRNN_wake_word().to(device)
model.load_state_dict(torch.load("JARVIS_najlepszy.pth", map_location=device))
model.eval()

model_com = CRNN_commands().to(device)
model_com.load_state_dict(torch.load("commands_test.pth", map_location=device))
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


            if jarvis_prob > 0.1: 
                commands_model()    # uruchom model komend 
            


def commands_model():
    audio = []
    i = 0
    while(i<30):
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

    commands_prediction_queue.put((switch_off, turn_on))
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
        self.setWindowTitle("Voice Recognition App")
        
        # 1. Tło okna - Ciemny gradient liniowy
        gradient = QLinearGradient(0, 0, 0, 800)
        gradient.setColorAt(0.0, QColor(17, 3, 104))    # Ciemny granat
        gradient.setColorAt(1.0, QColor(17, 19, 24))    # Bardzo ciemny, prawie czarny

        palette = self.palette()
        palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
        self.setPalette(palette)

        # Globalny styl CSS dla etykiet, aby były białe i czytelne na ciemnym tle
        self.setStyleSheet("""
            QLabel {
                color: #ffffff;
            }
        """)

        self.n = 0
        self.i = 0
        font = QFont("Segoe UI", 24)
        font.setBold(True)

        # STYLE DLA PRZYCISKÓW (zdefiniowany raz, aby nie powtarzać kodu)
        button_style = """
            QPushButton {
                background-color: #1a237e;      /* Ciemnoniebieskie tło */
                color: white;                   /* Biały tekst */
                border-radius: 15px;            /* Zaokrąglenie rogów */
                border: 2px solid #3f51b5;      /* Delikatna ramka */
            }
            QPushButton:hover {
                background-color: #283593;      /* Jaśniejszy po najechaniu */
            }
            QPushButton:pressed {
                background-color: #0f172a;      /* Ciemny po kliknięciu */
            }
        """

        # 2. Przycisk (Start / Stop)
        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.klikniecie_przycisku)
        self.button.setFixedSize(300, 100)
        self.button.setFont(font)
        self.button.setStyleSheet(button_style)

        # PRZYCISK EXIT (Podpięcie wbudowanej metody self.close)
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close) # <--- NADANIE FUNKCJI ZAMYKANIA
        self.exit_button.setFixedSize(300, 100)
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet(button_style)


        # 4. Zaokrąglona ramka (Kontener) dla wykresu
        self.plot_container = QFrame()
        self.plot_container.setStyleSheet("""
            QFrame {
                background-color: #171040;      /* Kolor tła wykresu */
                border-radius: 20px;            /* Zaokrąglenie rogów kontenera */
                border: 1px solid #33FFFF;      /* Obwódka */
            }
        """)

        # 5. Wykres (Przezroczyste tło, bo siedzi w zaokrąglonej ramce)
        self.plot_widget = pyqtgraph.PlotWidget()
        self.plot_widget.setBackground('transparent') 
        self.plot_widget.showAxis('left', False)
        self.plot_widget.showAxis('bottom', False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.setXRange(0, 16000)
        self.plot_widget.setAntialiasing(True)

        self.curve = self.plot_widget.plot(pen=pyqtgraph.mkPen(color=(0, 200, 255), width=2))

        # Wrzucamy wykres do wnętrza zaokrąglonej ramki (kontenera)
        container_layout = QVBoxLayout(self.plot_container)
        container_layout.setContentsMargins(10, 10, 10, 10) 
        container_layout.addWidget(self.plot_widget)  

        # 6. Licznik czasu (Timer) do odświeżania wykresu
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # 7. Budowanie głównej struktury layoutów okna
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # === GÓRA ===
        main_layout.addWidget(self.plot_container, stretch=5)
        
        # === DÓŁ: Nowy, uporządkowany układ symetryczny ===
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(20, 10, 20, 20) # Marginesy na dole okna

        # LEWA STRONA (Blok ze statusem i przyciskiem Start)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.button)

        # ŚRODEK (Tylko napis JARVIS wyśrodkowany idealnie)
        self.model_label = QLabel("JARVIS is NOT LISTENING")
        self.model_label.setFont(font)
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Środkowanie tekstu w widgetu

        # PRAWA STRONA (Blok z przyciskiem Exit - pusta etykieta u góry dla idealnej symetrii do statusu z lewej)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.exit_button)

        # Składanie wszystkiego w poziomy dolny pasek z automatycznym rozpychaniem (stretch)
        bottom_layout.addLayout(left_layout, stretch=1)
        bottom_layout.addWidget(self.model_label, stretch=2) # Środek dostaje więcej przestrzeni
        bottom_layout.addLayout(right_layout, stretch=1)

        main_layout.addLayout(bottom_layout, stretch=1)
        

    def klikniecie_przycisku(self):
        if self.timer.isActive():
            self.timer.stop()
            self.model_label.setText("JARVIS is NOT LISTENING")
            self.button.setText("Start")
            self.curve.setData([])
        else:
            self.timer.start(100)
            self.button.setText("Stop")


    def update_plot(self):
        # 1. Pobranie danych o wykresie i wybudzaniu (Wake Word)
        temp = None
        while not prediction_queue.empty():
            temp = prediction_queue.get_nowait()

        # Jeśli nie ma nowego audio, nie robimy nic z wykresem
        if temp is not None:
            audio, pred = temp
            audio = np.array(audio)
            self.curve.setData(audio, skipFiniteCheck=True)
            
            # Zapisujemy predykcję w pamięci obiektu, aby pamiętać stan między klatkami
            self.last_pred = pred 
        else:
            # Jeśli brak nowej paczki, przyjmij poprzedni stan lub 0
            if not hasattr(self, 'last_pred'):
                self.last_pred = 0

        # 2. Pobranie danych o komendach (Command Word) - NIE PRZERYWAMY SEKCJI WYŻEJ!
        temp_1 = None
        while not commands_prediction_queue.empty():
            temp_1 = commands_prediction_queue.get_nowait()

        # 3. Logika wyświetlania tekstów na podstawie stanu maszynowego
        if temp_1 is not None:
            self.curve.setPen(pyqtgraph.mkPen(color=(0, 200, 255), width=2))
            self.plot_container.setStyleSheet("""
            QFrame {
                background-color: #171040;      /* Kolor tła wykresu */
                border-radius: 20px;            /* Zaokrąglenie rogów kontenera */
                border: 1px solid #33FFFF;      /* Obwódka */
            }
        """)

            # JEŚLI PRZYSZŁA NOWA KOMENDA (Wyświetlamy ją na stałe i zatrzymujemy na chwilę animację listening)
            switch_off, turn_on = temp_1
            if switch_off > turn_on:
                self.model_label.setStyleSheet("color: #ff3333;") # Opcjonalnie: czerwony tekst dla off
                self.model_label.setText("Leds have been switched off")
            elif switch_off < turn_on:
                self.model_label.setStyleSheet("color: #33ff33;") # Opcjonalnie: zielony tekst dla on
                self.model_label.setText("Leds have been turned on")
            else:
                self.model_label.setStyleSheet("color: #FF8000;")
                self.model_label.setText("JARVIS didn't understand your command")
            
            # Tworzymy licznik zamrożenia ekranu komendy (np. na 25 klatek = 2.5 sekundy)
            self.display_freeze_counter = 10 


        else:
            # Sprawdzamy czy okno komunikatu o komendzie powinno jeszcze wisieć
            if hasattr(self, 'display_freeze_counter') and self.display_freeze_counter > 0:
                self.display_freeze_counter -= 1
                return # Zostawiamy aktualny tekst komendy na ekranie, ignorujemy dalszą animację
            
            # Standardowe zachowanie w zależności od detekcji Jarvis
            if self.last_pred > 0.1:
                # Przywracamy domyślny biały kolor tekstu po zniknięciu komunikatu komendy
                self.model_label.setStyleSheet("color: #ffffff;")
                self.curve.setPen(pyqtgraph.mkPen(color=(255, 10, 10), width=2))
                self.plot_container.setStyleSheet("""
            QFrame {
                background-color: #171040;      /* Kolor tła wykresu */
                border-radius: 20px;            /* Zaokrąglenie rogów kontenera */
                border: 1px solid #FF0000;      /* Obwódka */
            }
        """)
                
                self.model_label.setText("Issue your COMMAND ...")
            else:
                # Przywracamy domyślny biały kolor tekstu po zniknięciu komunikatu komendy
                self.model_label.setStyleSheet("color: #ffffff;")
                self.curve.setPen(pyqtgraph.mkPen(color=(0, 200, 255), width=2))
                self.plot_container.setStyleSheet("""
            QFrame {
                background-color: #171040;      /* Kolor tła wykresu */
                border-radius: 20px;            /* Zaokrąglenie rogów kontenera */
                border: 1px solid #33FFFF;      /* Obwódka */
            }
        """)
                
                # Animacja "LISTENING ...", wykonuje się TYLKO gdy nie ma wykrytej komendy
                if self.i % 4 == 0:
                    self.n += 1
                    dots = "." * (self.n % 4)
                    self.model_label.setText(f"JARVIS is LISTENING {dots}")
            
        self.i += 1


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()



# Sens będzie taki, że jeśli przycisk zostanie wciśnięty, to wyświetli się nam grafika z pyqtgraph, na której będą się wyświetlać właśnie przykładowe randomowe audio co 0.1 s


