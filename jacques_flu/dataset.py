from pathlib import Path
import typer
from data_pipeline.loader import FluDataLoader


from jacques_flu.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "flu_dat.csv",
    # ----------------------------------------------
):
    # Read in data
   

    fdl = FluDataLoader(input_path)

    df = fdl.load_data(hhs_kwargs={'rates': True})


    # Save data
    df.to_csv(output_path, index=False)    
    # -----------------------------------------


if __name__ == "__main__":
    app()
