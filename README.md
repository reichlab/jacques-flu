# Jacques Manuscript

A application of Jacques method to Flu forecasting.

## Running the project

## Virtual environment

Prerequisites:

- uv

Directions for setting up virtual environment:

1. Clone this repository

2. CHange to the repo's root directory:

`cd jacques-flu`

3. Create a Python virtual environment and install dependencies. THe command below creates an environment in the `.venv` directory, install Python if needed, installs project dependencies, and installs the package in editable mode:

`uv sync`

To do: Add pytest functionality


## Make commands
    This project is set up to allow for command line `make` commands

    - Produce flu data set from collection of raw data: `make data`
    - Create featurized data set: `make features`


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         jacques_flu and configuration for tools like black
│
│
├── artifacts          <- contains tables and figures to be used in manuscript
│   └── figures        <- Generated graphics and figures to be used in manuscript
|   └── tables         <- Generated tables to be used in manuscript
│
├── requirements       <- Folder containing requirement.txt files.
│
├── setup.cfg          <- Configuration file for flake8
│
└── jacques_flu   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes jacques_flu a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Project to do:

    - [] Train/Test set split
    - [] Fit Jacques to Flu Data