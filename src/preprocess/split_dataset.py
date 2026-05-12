"""
CMMD Split Dataset Script

Takes the clean CMMD dataset (e.g., `dataset_clean.csv`)
and:
- Groups by patient.
- Performs stratified split (preserving benign/malignant ratio).
- Adds a `split` column (values: "train", "val", "test").
- Saves back to CSV.

Recommended proportions:
    train: 70% of patients
    val  : 15%
    test : 15%
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv: str = "data/processed/dataset_clean.csv",
    output_csv: str = "data/processed/dataset_split.csv",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
):
    """
    Split CMMD data at the patient level with stratified train/val/test.

    Args:
        input_csv: Path to the input CSV (e.g., build_dataset output).
        output_csv: Path where the CSV with `split` column will be saved.
        train_frac: Fraction of patients used for training.
        val_frac: Fraction used for validation (rest goes to test).
        random_state: Seed for reproducibility.
    """
    print("=== CMMD SPLIT DATASET ===")
    print(f"Input: {input_csv}")

    # Read dataset
    df = pd.read_csv(input_csv)

    # Ensure patient_id and label are clean
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # --- 1. Patient-level information ---
    patients = (
        df[["patient_id", "label"]].drop_duplicates().reset_index(drop=True)
    )
    n_patients = len(patients)

    print("Total unique patients:", n_patients)
    print("Label distribution (all patients):")
    print(patients["label"].value_counts())

    # --- 2. Define split sizes ---
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, f"test_frac = {test_frac:.2f} must be > 0"

    # --- 3. Stratified train / (rest)
    idx_train, idx_rest = train_test_split(
        patients.index,
        stratify=patients["label"],
        test_size=(1.0 - train_frac),
        random_state=random_state,
    )

    # Split rest into val and test
    idx_val, idx_test = train_test_split(
        idx_rest,
        stratify=patients.loc[idx_rest, "label"],
        test_size=test_frac / (val_frac + test_frac),
        random_state=random_state,
    )

    # Assign splits to patient IDs
    patient2split = {}
    for idx in idx_train:
        patient2split[patients.loc[idx, "patient_id"]] = "train"
    for idx in idx_val:
        patient2split[patients.loc[idx, "patient_id"]] = "val"
    for idx in idx_test:
        patient2split[patients.loc[idx, "patient_id"]] = "test"

    # --- 4. Map split to full dataset (per image, but consistent per patient)
    df["split"] = df["patient_id"].map(patient2split)

    # Sanity: all rows should have a split
    assert (
        df["split"].isna().sum() == 0
    ), "Some patients did not get a split"

    # --- 5. Print final statistics ---
    print("\nSplit overview (patient‑count):")
    pt_split = df[["patient_id", "split"]].drop_duplicates()
    print(pt_split["split"].value_counts())

    print("\nPer‑split label distribution (patient‑level):")
    for s in ["train", "val", "test"]:
        sub = pt_split[pt_split["split"] == s]
        df_sub = df[df["patient_id"].isin(sub["patient_id"].values)]
        print(f"{s:6s} (patients):", len(sub))
        print(f"{s:6s} labels:", df_sub["label"].value_counts())

    # --- 6. Save to disk ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDataset with `split` column saved to: {output_csv}")

    return df


if __name__ == "__main__":
    split_dataset()
