"""
Step 1 of Part C.

The NIH ChestX-ray14 dataset gives labels as a single text column, e.g.
"Cardiomegaly|Effusion" or "No Finding". This script turns that into a
proper multi-label format: one column per pathology, 1 if present, 0 if
not - which is what a PyTorch dataloader needs to train a classifier.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ---- LOCAL path (default) ----
RAW_CSV = BASE_DIR / "data" / "Data_Entry_2017.csv"

# ---- KAGGLE path (use this instead if running in a Kaggle notebook) ----
# RAW_CSV = Path("/kaggle/input/nih-chest-xrays/data/Data_Entry_2017.csv")

OUTPUT_CSV = BASE_DIR / "data" / "labels_multilabel.csv"

# the 14 official NIH pathology classes (excluding "No Finding")
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def main():
    df = pd.read_csv(RAW_CSV)

    for pathology in PATHOLOGIES:
        # 1 if this pathology's name appears in the "Finding Labels" text, else 0
        df[pathology] = df["Finding Labels"].apply(lambda labels: int(pathology in labels))

    output_columns = ["Image Index"] + PATHOLOGIES
    df[output_columns].to_csv(OUTPUT_CSV, index=False)

    print(f"done. wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
