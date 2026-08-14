"""
Downloads the RSNA dataset onto your own machine using the Kaggle API.

Only needed if running LOCALLY. On Kaggle itself, skip this file entirely -
the dataset is already attached to the notebook automatically, no download
needed.

One-time setup before running this:
1. pip install kaggle
2. Go to kaggle.com -> your profile -> Settings -> API -> "Create New Token"
   This downloads a file called kaggle.json
3. Put that file at:
   - Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
   - Mac/Linux: ~/.kaggle/kaggle.json
4. Join the competition once (accept rules) at:
   kaggle.com/c/rsna-pneumonia-detection-challenge
"""

import subprocess
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "rsna-pneumonia-detection-challenge.zip"

COMPETITION = "rsna-pneumonia-detection-challenge"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("downloading dataset (this is a few GB, may take a while)...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        check=True,
    )

    print("unzipping...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("done. data is in:", DATA_DIR)


if __name__ == "__main__":
    main()
