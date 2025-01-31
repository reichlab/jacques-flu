from pathlib import Path
import polars as pl
import tensorflow as tf
from jacques import kcqe
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import typer
from loguru import logger
from tqdm import tqdm

from jacques_flu.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# def fit_jacques(
#         x_train: pd.DataFrame, 
#         y_train: pd.DataFrame, 
#         x_kernel="gaussian_diag", 
#         tau = tf.constant(np.array([0.1, 0.5, 0.9]), dtype=tf.float32)):
    
#     """
#     Fit a Jacques model to the data
    
#     Inputs:
#         x_train: pandas DataFrame with features training data
#         y_train: pandas DataFrame with target training data
#         x_kernel: kernel function for x
#         tau: tensor with quantile levels to estimate
    
#     Outputs:
#         Jacques model
#     """
    
#     # This could be fed as function input
#     tau = tf.constant(np.array([0.1, 0.5, 0.9]), dtype=tf.float32)

#     kcqe_obj = kcqe.KCQE(x_kernel = x_kernel, p= x_train.shape[1])

#     block_size = 21

#     num_blocks = math.floor(y_train.shape[1] / block_size)
    
#     jacques_generator = kcqe_obj.generator(
#         x_train_val = x_train,
#         y_train_val = y_train,
#         batch_size= num_blocks,
#         block_size = block_size,
#     )

#     init_param_vec = tf.constant(np.zeros(kcqe_obj.n_param), dtype=np.float32)

#     breakpoint() # This breakpoint will be hit when the function is called
#     param_vec = kcqe_obj.fit(xval_batch_gen = jacques_generator,
#             num_blocks = num_blocks,
#             tau = tau,
#             optim_method = "adam",
#             num_epochs = 10,
#             learning_rate = 0.1,
#             init_param_vec = init_param_vec,
#             verbose=True)
    
#     return param_vec, kcqe_obj

# -------------------------
# Define custom dataset for PyTorch
# -------------------------
class CustomDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features.values, dtype = torch.float32)
        self.labels = torch.tensor(labels.values, dtype = torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]



# -------------------------
#  Define Neural Network
# -------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size=45):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Assuming regression task
        )

    def forward(self, x):
        return self.model(x)
    
def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    features_list_path: Path = PROCESSED_DATA_DIR / "feature_list.txt",
    labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    model_path: Path = MODELS_DIR / "trained_neural_net.pth",
    # -----------------------------------------
):
    # Tuning Parameters
    batch_size = 32
    input_size = 43  # Number of features



    # Features list
    with open(features_list_path) as file:
        features_list = [line.rstrip() for line in file]

    # Read in training data
    train_features = pl.read_csv(features_path).to_pandas()
    train_features = train_features[features_list]

    # Read in labels
    train_labels = pl.read_csv(labels_path).to_pandas()

    train_dataset = CustomDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_size)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)

    learning_rate = 0.001
   
    criterion = nn.MSELoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    num_epochs = 10

    #Train the model
    logger.info("Training Neural Network model...")
    for epoch in tqdm(range(num_epochs), total=10):
        train(train_loader, model, criterion, optimizer, device)
    logger.success("Modeling training complete.")
    
    # Save model to disk
    torch.save(model.state_dict(), model_path)
    print("Model saved to trained_model.pth")

if __name__ == "__main__":
    app()
