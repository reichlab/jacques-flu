from pathlib import Path
import typer
import pandas as pd
from jacques_flu.config import HHS_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = HHS_DATA_DIR / "flu_dat.csv",
    # ----------------------------------------------
):
    # Read in data from FluSight-forecast-hub
    url = 'https://github.com/cdcepi/FluSight-forecast-hub/blob/main/target-data/target-hospital-admissions.csv'
    df = pd.read_csv(url, index_col = 0)


    # Save data
    df.to_csv(output_path, index=False)     
    # -----------------------------------------


if __name__ == "__main__":
    app()