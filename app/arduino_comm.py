# this file will communicate the output of main.py file with arduino

import serial


ser = serial.Serial('/dev/ttyACM0', 9600)  


while True:
    decision = int(input("Wpisz '1' w celu włączenia ledów, albo wpisz '0' w celu wyłączenia ledów: "))

    if decision == 1:
        ser.write(b'1')  # włącz LED
    elif decision == 0:
        ser.write(b'0')  # wyłącz LED
    else:
        break
