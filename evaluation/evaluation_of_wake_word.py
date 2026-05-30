# in this file we will be evaluating our model for wake word recognition on our own recordings

from evaluation_datasets_import import load_JARVIS_records_for_evaluation#, load_background_onesec_records_for_evaluation


if __name__ == "__main__":
    
    JARVIS = load_JARVIS_records_for_evaluation()

    print("Length of JARVI for evaluation:", len(JARVIS))
