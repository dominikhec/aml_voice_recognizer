# in this file we will be training command_recognition_model on our data


#import torchaudio
from google_speech_commands_dataset import load_twosec_google_speech_commands_for_training
from urban_sounds_dataset import load_twosec_urban_sounds_for_training
from esc50_dataset import load_twosec_esc50_for_training
from our_records_dataset import load_switch_off_records_for_training, load_turn_on_records_for_training

if __name__ == "__main__":
    
    google_dataset = load_twosec_google_speech_commands_for_training()
    urban_sounds_dataset = load_twosec_urban_sounds_for_training()
    esc50_dataset = load_twosec_esc50_for_training()
    turn_on_dataset = load_turn_on_records_for_training()
    switch_off_dataset = load_switch_off_records_for_training()

    print("Length of Google dataset for training:", len(google_dataset))
    print("Length of Urban sounds dataset for training:", len(urban_sounds_dataset))
    print("Length of ESC50 dataset for training:", len(esc50_dataset))
    print("Length of Turn on dataset for training:", len(turn_on_dataset))
    print("Length of Switch off dataset for training:", len(switch_off_dataset))




