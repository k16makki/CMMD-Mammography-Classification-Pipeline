"""
CMMD Build Dataset Script

Takes the raw `dataset.csv` (from `cmmd_api_download`) and:
- Ensures consistency in paths and labels.
- Adds basic metadata useful for training (e.g., image view side).
- Verifies patient‑level structure and label sanity.

Note:
    Assumes `dataset.csv` is at:
        data/processed/dataset.csv
"""

import os
import pandas as pd
from pathlib import Path


def build_dataset(
    input_csv: str = "data/processed/dataset.csv",
    output_csv: str = "data/processed/dataset_clean.csv",
):
    """
    Build a clean, consistent dataset ready for deep‑learning.

    Args:
        input_csv: Path to the raw `dataset.csv` (from `cmmd_api_download`).
        output_csv: Path where the cleaned CSV will be saved.
    """
    df = pd.read_csv(input_csv)

    print("=== CMMD BUILD DATASET ===")
    print("Input dataset shape:", df.shape)
    print("Label distribution (before cleaning):")
    print(df["label"].value_counts())

    # --- 1. Sanity: drop malformed rows ---
    df = df.dropna(subset=["path", "patient_id"])
    df = df[df["path"].str.strip() != ""]

    # Ensure patient_id is str (important for grouping/splitting)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    # --- 2. Keep only valid binary labels ---
    df = df[df["label"].isin([0, 1])]

    # --- 3. Preserve side/view if present, otherwise create them ---
    if "side" not in df.columns:
        df["side"] = "unknown"
    else:
        df["side"] = df["side"].astype(str).str.strip().str.upper()

    if "view" not in df.columns:
        df["view"] = "unknown"
    else:
        df["view"] = df["view"].astype(str).str.strip().str.upper()

    # --- 4. Patient-level sanity ---
    n_patients = df["patient_id"].nunique()
    print("Unique patients:", n_patients)
    print("Label distribution (after cleaning):")
    print(df["label"].value_counts())

    # Optionnel: un petit check side
    print("Side distribution:", df["side"].value_counts())

    # --- 5. Save clean dataset ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Cleaned dataset saved to:", output_csv)

    return df


if __name__ == "__main__":
    # You can run this directly from the project root
    build_dataset()
