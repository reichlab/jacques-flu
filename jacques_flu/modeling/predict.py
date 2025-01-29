from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import typer
from loguru import logger
from tqdm import tqdm

from jacques_flu.config import MODELS_DIR, PROCESSED_DATA_DIR
from jacques_flu.modeling.train import CustomDataset, NeuralNet

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "trained_neural_net.pth",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    features_list_path: Path = PROCESSED_DATA_DIR / "feature_list.txt",
    # -----------------------------------------
):
    input_size = 43 

    with open(features_list_path) as file:
        features_list = [line.rstrip() for line in file]

    # Read in training data
    test_features = pl.read_csv(features_path).to_pandas()
    test_features = test_features[features_list]
    #test_features = torch.tensor(test_features.values, dtype = torch.float32)

    # Read in labels
    test_labels = pl.read_csv(labels_path).to_pandas()

    model = NeuralNet(input_size)
    model.load_state_dict(torch.load(model_path))


    model = torch.load(model_path)

    test_dataset = CustomDataset(test_features, test_labels)

    predictions = model(test_dataset.features)

    # access Variable's tensor, copy back to CPU, convert to numpy
    arr = predictions.data.cpu().numpy()
    # write CSV
    np.savetxt(predictions_path, arr)
    # -----------------------------------------
    print("Predictions saved to test_predictions.csv")

    
if __name__ == "__main__":
    app()
