# in this file we will be evaluating our model for wake word recognition on our own recordings

import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

from evaluation_datasets_import import load_JARVIS_records_for_evaluation, load_background_onesec_records_for_evaluation
from models.wake_word_model import *
import torch
import random
from torch.utils.data import DataLoader


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    JARVIS = load_JARVIS_records_for_evaluation()
    background = load_background_onesec_records_for_evaluation()

    background = background[:100]  # we take only 100 background samples for evaluation to balance the dataset

    #print("Length of JARVI for evaluation:", len(JARVIS))
    #print("Length of background records for evaluation:", len(background))

    #model = SimpleCNN()
    model = CRNN()
    model.load_state_dict(torch.load("wake_word_model.pth"))
    model.eval()

    test_data = JARVIS + background

    random.shuffle(test_data)

    test_loader = DataLoader(test_data, shuffle = False)

    correct = 0
    total = 0
    accuracy = 0.0
    i = 0

    with torch.no_grad(): # disables gradient tracking
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            print(f"{i+1}: Predicted: {predicted.item()}, Actual: {target.item()}")
            i += 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.5f}%")







