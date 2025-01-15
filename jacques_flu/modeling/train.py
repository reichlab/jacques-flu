from pathlib import Path
import polars as pl
import tensorflow as tf
from jacques import kcqe
import numpy as np
import math
import pandas as pd

import typer
from loguru import logger
from tqdm import tqdm

from jacques_manuscript.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def fit_jacques(
        x_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        x_kernel="gaussian_diag", 
        tau = tf.constant(np.array([0.1, 0.5, 0.9]), dtype=tf.float32)):
    
    """
    Fit a Jacques model to the data
    
    Inputs:
        x_train: pandas DataFrame with features training data
        y_train: pandas DataFrame with target training data
        x_kernel: kernel function for x
        tau: tensor with quantile levels to estimate
    
    Outputs:
        Jacques model
    """
    
    # This could be fed as function input
    tau = tf.constant(np.array([0.1, 0.5, 0.9]), dtype=tf.float32)

    kcqe_obj = kcqe.KCQE(x_kernel = x_kernel, p= x_train.shape[1])

    block_size = 21

    num_blocks = math.floor(y_train.shape[1] / block_size)
    
    jacques_generator = kcqe_obj.generator(
        x_train_val = x_train,
        y_train_val = y_train,
        batch_size= num_blocks,
        block_size = block_size,
    )

    init_param_vec = tf.constant(np.zeros(kcqe_obj.n_param), dtype=np.float32)

    breakpoint() # This breakpoint will be hit when the function is called
    param_vec = kcqe_obj.fit(xval_batch_gen = jacques_generator,
            num_blocks = num_blocks,
            tau = tau,
            optim_method = "adam",
            num_epochs = 10,
            learning_rate = 0.1,
            init_param_vec = init_param_vec,
            verbose=True)
    
    return param_vec, kcqe_obj


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    # Read in training data
    x_train = pl.read_csv(features_path).to_pandas()

    # Read in labels
    y_train = pl.read_csv(labels_path).to_pandas()


    fit_jacques(x_train, y_train)


    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
