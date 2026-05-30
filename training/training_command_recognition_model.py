# in this file we will be training command_recognition_model on our data


#import torchaudio
from training.onesec_google_speech_commands_dataset import load_google_speech_commands_for_training
from training.onesec_urban_sounds_dataset import load_urban_sounds_for_training
from training.onesec_esc50_dataset import load_esc50_for_training

if __name__ == "__main__":
    
    google_dataset = load_google_speech_commands_for_training()
    urban_sounds_dataset = load_urban_sounds_for_training()
    esc50_dataset = load_esc50_for_training()

    print("Length of Google dataset for training:", len(google_dataset))
    print("Length of Urban sounds dataset for training:", len(urban_sounds_dataset))
    print("Length of ESC50 dataset for training:", len(esc50_dataset))




