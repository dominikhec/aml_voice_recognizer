# JARVIS Voice Recognizer

This project aims to create a Machine Learning model that recognizes the wake-word "JARVIS" and then processes two specific voice commands: "Turn on the LEDs" and "Switch off the LEDs".

The only folder that is not included in this public repository is the "data" folder. It has been excluded for the following reasons:
 - Privacy: It contains personal voice recordings with private information.
 - Size: The dataset exceeds 10 GB.

If you want to see the model working for yourself, you have two options:

1. You have an Arduino board connected with the code provided in the "arduino" folder:
Then you can run the main script located in the "app" directory by typing in the bash terminal: `python app/main.py`

2. You don't have an Arduino board connected:
 - Firstly, you need to comment out the following lines of code in "main.py":
    - line 42
    - from line 135 to line 140
 - Then you can run the main script located in the "app" directory by typing in the bash terminal: `python app/main.py`