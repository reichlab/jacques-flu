from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datetime import timedelta
import polars as pl
import plotly.express as px

from jacques_flu.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


#Visualization Function

def plot_time_series(df, target, max_date, plot_weeks, time_var, group_var, group_value):
    """
    Plot a time series of the target variable for a given group.

    Parameters
    ---------- 
    df : pd.DataFrame
        Data to plot.
    target : str
        Name of the target variable.
    max_date : datetime
        Latest date to plot.
    plot_weeks : int
        Number of weeks to plot.
    time_var : str  
        Name of the time variable.
    group_var : str 
        Name of the group variable.
    group_value : str
        Value of the group variable to plot.

    Returns
    ------- 
    Plotly figure

    To Do: 
        - Add default values
        - What to do if there aren't enough weeks in the test set
        - Allow for multiple group values (to accomodate different sources)
    """

    min_date = max_date - timedelta(weeks=plot_weeks)

    # Filter the data for the given state and date range
    state_data = (df.filter(pl.col('source') == 'hhs')
                .filter(pl.col(group_var) == group_value)
                .filter(pl.col(time_var) >= min_date)
                .filter(pl.col(time_var) <= max_date)
                .select([time_var, target])
                .sort(pl.col(time_var)))
    
    # Create a time series plot using Plotly
    fig = px.line(state_data, x='week', y=target, title=f'Flu Counts in {group_value} from {min_date} to {max_date}')
    
    # Show the plot
    fig.show()