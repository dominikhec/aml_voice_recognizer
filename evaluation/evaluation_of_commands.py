# in this file we will be evaluating our model for commands recognition on our own recordings

from evaluation_datasets_import import load_switch_off_records_for_evaluation, load_turn_on_records_for_evaluation#, load_background_twosec_records_for_evaluation


if __name__ == "__main__":
    
    switch_off = load_switch_off_records_for_evaluation()
    turn_on = load_turn_on_records_for_evaluation()

    print("Length of switch off records for evaluation:", len(switch_off))
    print("Length of turn on records for evaluation:", len(turn_on))

