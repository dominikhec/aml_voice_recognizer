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
    
    # Powinienem najpierw pobrać, następnie zamienić na mel spectogramy, popodpisywać, potasować i wtedy dopiero rozdzielić na trening i validację. 
    # Następnie należy zrobić augmentację nagrań JARVIS i wtedy trzeba przetasować jeszcze raz

    # Należy pobrać:
    # 330 własnych nagrań (JARVIS)
    # 1650 nagrań ciszy
    # 6930 nagrań z Google Speech Commands
    # 550 nagrań z Urban Sounds
    # 550 nagrań z ESC50
    # 1100 nagrań pokoju jak się coś dzieje


    # Struktura danych:

    # Do trenowania modelu

    # 300 nagrań własnych (JARVIS), z których za pomocą augmentacji możemy zrobić 5400 nagrań (300 * 18 = 5400)
    # 1500 nagrań ciszy
    # 6300 nagrań z Google Speech Commands
    # 500 nagrań z Urban Sounds
    # 500 nagrań z ESC50 
    # 1000 nagrań pokoju jak się coś dzieje


    # Do validacji modelu

    # 30 własnych nagrań zostawić do validacji na koniec
    # 150 nagran ciszy
    # 630 nagrań z Google Speech Commands
    # 50 nagrań z urban sounds
    # 50 nagrań z ESC50
    # 100 nagrań pokoju jak się coś dzieje


    print("Length of Google dataset for training:", len(google_dataset))
    print("Length of Urban sounds dataset for training:", len(urban_sounds_dataset))
    print("Length of ESC50 dataset for training:", len(esc50_dataset))
    print("Length of JARVIS dataset for training:", len(JARVIS_dataset))

