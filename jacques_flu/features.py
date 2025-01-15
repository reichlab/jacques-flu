from pathlib import Path
import fnmatch

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timeseriesutils import featurize
from data_pipeline.utils import get_holidays
import typer

from jacques_flu.config import PROCESSED_DATA_DIR

app = typer.Typer()





def create_features_and_targets(df, incl_level_feats, max_horizon, curr_feat_names = []):
    '''
    Create features and targets for prediction
    
    Parameters
    ----------
    df: pandas dataframe
      data frame with data to "featurize"
    incl_level_feats: boolean
      include features that are a measure of local level of the signal?
    max_horizon: int
      maximum forecast horizon
    curr_feat_names: list of strings
      list of names of columns in `df` containing existing features
    
    Returns
    -------
    tuple with:
    - the input data frame, augmented with additional columns with feature and
      target values
    - a list of all feature names, columns in the data frame
    '''
    
    # current features; will be updated
    feat_names = curr_feat_names
    
    # # one-hot encodings of data source, agg_level, and location
    # for c in ['source', 'agg_level', 'location']:
    #     ohe = pd.get_dummies(df[c], prefix=c)
    #     df = pd.concat([df, ohe], axis=1)
    #     feat_names = feat_names + list(ohe.columns)
    
    # season week relative to christmas
    df = df.merge(
            get_holidays() \
                .query("holiday == 'Christmas Day'") \
                .drop(columns=['holiday', 'date']) \
                .rename(columns={'season_week': 'xmas_week'}),
            how='left',
            on='season') \
        .assign(delta_xmas = lambda x: x['season_week'] - x['xmas_week'])
    
    feat_names = feat_names + ['delta_xmas']
    
    # features summarizing data within each combination of source and location
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_trans_cs',
                    'taylor_degree': 2,
                    'window_align': 'trailing',
                    'window_size': [4, 6],
                    'fill_edges': False
                }
            },
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_trans_cs',
                    'taylor_degree': 1,
                    'window_align': 'trailing',
                    'window_size': [3, 5],
                    'fill_edges': False
                }
            },
            {
                'fun': 'rollmean',
                'args': {
                    'columns': 'inc_trans_cs',
                    'group_columns': ['location'],
                    'window_size': [2, 4]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'lag',
                'args': {
                    'columns': ['inc_trans_cs'] + new_feat_names,
                    'lags': [1, 2]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # add forecast targets
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'horizon_targets',
                'args': {
                    'columns': 'inc_trans_cs',
                    'horizons': [(i + 1) for i in range(max_horizon)]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # we will model the differences between the prediction target and the most
    # recent observed value
    df['delta_target'] = df['inc_trans_cs_target'] - df['inc_trans_cs']
    
    # if requested, drop features that involve absolute level
    if not incl_level_feats:
        feat_names = _drop_level_feats(feat_names)
    
    return df, feat_names


def _drop_level_feats(feat_names):
    level_feats = ['inc_trans_cs', 'inc_trans_cs_lag1', 'inc_trans_cs_lag2'] + \
                  fnmatch.filter(feat_names, '*taylor_d?_c0*') + \
                  fnmatch.filter(feat_names, '*inc_trans_cs_rollmean*')
    feat_names = [f for f in feat_names if f not in level_feats]
    return feat_names



## Split the data into training and testing sets
def train_test_split(df, date_col, horizon_weeks, forecast_date):
    """
    Split data into training and testing sets based on a date of interest and a horizon.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to split.
    date_col : str
        Name of the column containing the date.
    horizon : int
        Number of weeks to forecast.
    date_of_interest : str
        Date to split the data on.
        
    Returns
    -------
    train_df = pd.DataFrame
        Training data.
    test_df = pd.DataFrame
        Testing data.

    To Do: Add default values, or what to do if there aren't enough weeks in the test set
    """

    if isinstance(forecast_date, str):
        forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').date()


    max_date = forecast_date + timedelta(weeks=horizon_weeks)
    # Convert date column to datetime
    #df = df.with_columns(pl.col(date_col).str.to_date())
    
    # Split data
    train_df = df.filter(pl.col(date_col) < forecast_date)
    
    test_df = df.filter((pl.col(date_col) >= forecast_date) & (pl.col(date_col) < max_date))
    
    return train_df, test_df

# Slplit data into features and labels
def split_features_labels(df, target_col, features_list):
    """
    Split data into features and labels
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to split.
    target_col : str
        Name of the column containing the target variable.
        
    Returns
    -------
    features = pd.DataFrame
        Data frame with features.
    labels = pd.DataFrame
        Data frame with labels.
    """
    
    features = df.select(features_list)
    labels = df.select(target_col)
    
    return features, labels

@app.command()
def main(
    # ----------------------------------------
    input_path: Path = PROCESSED_DATA_DIR / "flu_dat.csv",
    output_path: Path = PROCESSED_DATA_DIR / "flu_features.csv",
    # -----------------------------------------
):
    df = pd.read_csv(input_path)

    df_feat, feat_names = create_features_and_targets(
        df = df,
        incl_level_feats=True,
        max_horizon=3,
        curr_feat_names=['inc_trans_cs', 'season_week', 'log_pop']
        )

    df_feat.dropna(inplace=True)

    df_feat = pl.from_pandas(df_feat, schema_overrides={'wk_end_date': pl.Date})

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df_feat, 'wk_end_date', 4, '2023-11-23')

    #Split training and testing into features and labels
    train_features, train_labels = split_features_labels(train_df, "delta_target", feat_names)
    test_features, test_labels = split_features_labels(test_df, "delta_target", feat_names)
    
    # with open(PROCESSED_DATA_DIR / "feature_list.txt", 'w') as f:
    #     for line in feat_names:
    #         f.write(f"{line}\n")
    
    df_feat.write_csv(output_path, separator=',')

    test_features.write_csv(PROCESSED_DATA_DIR / "test_features.csv", separator=',')
    test_labels.write_csv(PROCESSED_DATA_DIR / "test_labels.csv", separator=',')
    train_features.write_csv(PROCESSED_DATA_DIR / "train_features.csv", separator=',')
    train_labels.write_csv(PROCESSED_DATA_DIR / "train_labels.csv", separator=',')
    # -----------------------------------------


if __name__ == "__main__":
    app()


