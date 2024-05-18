import os, sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import json
from pathlib import Path

from src.misc import tensor_to_list, state_dict_to_json
from src.data import prepare_dataset
from src.model import MLP

def run_training(train_loader, model, num_epochs=30):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_and_save(model, data_test, labels_test, filename, debug=False):
    with torch.no_grad():
        if debug:
            raw_outputs, test_predictions = model(data_test.float(), debug=True)
            custom_print("Raw outputs:", raw_outputs)  # This will print the raw outputs for debugging
        else:
            test_predictions = model(data_test.float())

        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]
    
    print(f'Model Accuracy: {test_accuracy:.2f} \n')
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    
    with open(filename, 'w') as f:
        f.write(state_dict_json)



def train_model(df_path: Path, checkpoint_path: Path):
    data_train, data_calib, data_test, labels_train, labels_calib, labels_test, train_loader = prepare_dataset(df_path)

    # MLP 1-layer
    model = MLP(data_train.shape[1])
    trained_model = run_training(train_loader, model)
    evaluate_and_save(trained_model, data_test, labels_test, checkpoint_path)

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
            
    BASE = Path.cwd()
    data_path = BASE / "data/df.csv"
    checkpoint_path = BASE / "checkpoint/MLP.json"

    train_model(data_path, checkpoint_path)
