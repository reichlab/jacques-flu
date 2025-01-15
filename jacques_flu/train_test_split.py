import polars as pl
import pandas as pd
from datetime import timedelta, datetime

from pathlib import Path
from jacques_flu.config import PROCESSED_DATA_DIR
import typer

# Pass some date, split the data given some date and horizon

#The training data is everything before the date, the testing data is horizon number of weeks after 

app = typer.Typer()



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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR/ "flu_features.csv",
    output_path: Path = PROCESSED_DATA_DIR,
    # ----------------------------------------------
):
    # Read in data
   
    # Read in featured data
    df = pl.read_csv(input_path, schema_overrides={'location': pl.String, 'wk_end_date': pl.Date})

    # Split data on some date
    train_df, test_tf = train_test_split(df, 'wk_end_date', 4, '2023-11-23')


    # Save data
    train_df.write_csv(output_path/ "training_data.csv", separator=',')
    test_tf.write_csv(output_path/ "testing_data.csv", separator=',')    
    # -----------------------------------------


if __name__ == "__main__":
    app()
