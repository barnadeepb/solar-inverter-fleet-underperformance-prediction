"""Download the raw solar plant generation and weather CSVs.

Source: Kaggle dataset `anikannal/solar-power-generation-data`, mirrored as
plain CSV files on GitHub (Kaggle itself requires authenticated API access).
The raw files are also committed under data/raw/ so this script is only
needed to re-fetch or refresh them.
"""

import pathlib
import urllib.request

MIRROR_BASE = "https://raw.githubusercontent.com/kaivalpanchal/Solar-Panel-Power-Generation/main"

FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{MIRROR_BASE}/{name}"
        dest = RAW_DIR / name
        print(f"fetching {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    fetch_all()
