# in this file we will be training models on our data


#import torchaudio
from google_speech_commands_dataset import load_onesec_google_speech_commands_for_training
from urban_sounds_dataset import load_onesec_urban_sounds_for_training
from esc50_dataset import load_onesec_esc50_for_training
from our_records_dataset import load_JARVIS_records_for_training


if __name__ == "__main__":
    
    google_dataset = load_onesec_google_speech_commands_for_training()
    urban_sounds_dataset = load_onesec_urban_sounds_for_training()
    esc50_dataset = load_onesec_esc50_for_training()
    JARVIS_dataset = load_JARVIS_records_for_training()

    print("Length of Google dataset for training:", len(google_dataset))
    print("Length of Urban sounds dataset for training:", len(urban_sounds_dataset))
    print("Length of ESC50 dataset for training:", len(esc50_dataset))
    print("Length of JARVIS dataset for training:", len(JARVIS_dataset))

