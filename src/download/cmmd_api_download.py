"""
CMMD Automated Download Pipeline

Downloads CMMD DICOM series and clinical labels from TCIA in a
resumable, idempotent, and Docker‑friendly way.

Key features:
- Fetches series metadata from TCIA API and caches it to disk.
- Downloads DICOM series as .zip files, skipping those already present.
- Downloads the CMMD clinical Excel file directly from TCIA's static URL.
- Loads labels from the Excel file, aligns patient IDs, and maps labels to 0/1.
- Extracts DICOM paths and `PatientID` from the archives.
- Builds a unified CSV (`dataset.csv`) mapping DICOM paths to labels.

Arguments:
    --data_dir          Root data directory (e.g., /data in Docker).
    --collection        TCIA collection name (default: "CMMD").
    --max_cases         Number of series to download (default: 20).
    --force_download    Whether to redownload images and clinical file even if they exist.
"""

import os
import requests
import pandas as pd
import zipfile
import pydicom
from tqdm import tqdm
from argparse import ArgumentParser


# Base TCIA API endpoint for series/image download
BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_series(
    collection: str = "CMMD", cache_dir: str = "data/raw/CMMD"
) -> pd.DataFrame:
    """
    Fetch CMMD series metadata from TCIA.

    Args:
        collection: TCIA collection name (e.g., "CMMD").
        cache_dir: Directory where `series.csv` will be cached.

    Returns:
        DataFrame with series metadata (cached if already exists).
    """
    cache_path = os.path.join(cache_dir, "series.csv")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print("→ Using cached metadata")
        return pd.read_csv(cache_path)

    print("→ Fetching metadata from TCIA...")
    url = f"{BASE}/getSeries"

    r = requests.get(
        url,
        params={"Collection": collection},
        timeout=30,
    )
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    df.to_csv(cache_path, index=False)
    print("→ Metadata cached")

    return df


def download_series(
    series_uid: str, out_dir: str, force: bool = False
) -> str:
    """
    Download a DICOM series as a .zip file.

    Args:
        series_uid: SeriesInstanceUID from TCIA metadata.
        out_dir: Directory where zips are stored (e.g., "data/raw/CMMD/zips").
        force: If True, redownload even if file exists.

    Returns:
        Path to the downloaded .zip file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{series_uid}.zip")

    if os.path.exists(path) and not force:
        return path

    url = f"{BASE}/getImage"
    r = requests.get(
        url,
        params={"SeriesInstanceUID": series_uid},
        stream=True,
        timeout=60,
    )
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return path


def download_clinical(
    out_path: str, force: bool = False
) -> str:
    """
    Download the CMMD clinical XLSX file from TCIA's static URL.

    Args:
        out_path: Path where `clinical.xlsx` will be saved.
        force: If True, redownload even if file exists.

    Returns:
        Path to the downloaded clinical file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not force:
        return out_path

    # Direct static URL to the CMMD clinical Excel file
    url = "https://www.cancerimagingarchive.net/wp-content/uploads/CMMD_clinicaldata_revision.xlsx"
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    # Detect if we got HTML instead of the file
    ctype = r.headers.get("Content-Type", "")
    if "html" in ctype.lower():
        raise ValueError(f"Got HTML instead of Excel: {r.url}")

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("→ Clinical file downloaded correctly")
    return out_path


def load_labels(path: str) -> pd.DataFrame:
    """
    Load CMMD clinical labels and map them to 0/1, with side (L/R).

    The clinical file has:
        - `ID1` as the patient identifier.
        - `LeftRight` as the side of the breast ("L"/"R").
        - `classification` as the label column ("Benign"/"Malignant").
    """
    df = pd.read_excel(path, engine="openpyxl")

    # Normaliser les noms de colonnes
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    pid_candidates = ["id1", "patient_id", "patientid", "id"]
    side_candidates = ["leftright", "laterality", "side"]
    label_candidates = ["classification", "label", "diagnosis"]

    pid_col = next((c for c in pid_candidates if c in df.columns), None)
    side_col = next((c for c in side_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if pid_col is None or side_col is None or label_col is None:
        raise ValueError(
            f"Could not find patient, side or label columns in {df.columns.tolist()}"
        )

    df = df[[pid_col, side_col, label_col]].copy()
    df.columns = ["patient_id", "side", "label"]

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # On ne garde que L/R, le reste sera ignoré au moment du merge
    df = df[df["side"].isin(["L", "R"])]

    print("\nLabel distribution (raw clinical):")
    print(df["label"].value_counts())

    # Map to binary labels
    df["label"] = df["label"].map({"benign": 0, "malignant": 1})

    if df["label"].isna().sum() > 0:
        raise ValueError("Unexpected label values in clinical file")

    return df



def extract_images(zip_dir: str, extract_dir: str) -> pd.DataFrame:
    os.makedirs(extract_dir, exist_ok=True)
    rows = []

    for f in os.listdir(zip_dir):
        if not f.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, f)
        series_folder = os.path.join(extract_dir, f.replace(".zip", ""))

        if not os.path.exists(series_folder):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(series_folder)

        for root, _, files in os.walk(series_folder):
            for file in files:
                if not file.endswith(".dcm"):
                    continue

                path = os.path.join(root, file)

                try:
                    dcm = pydicom.dcmread(path, stop_before_pixels=True)

                    pid = getattr(dcm, "PatientID", None)
                    if not pid:
                        continue

                    side = getattr(dcm, "ImageLaterality", None) or getattr(dcm, "Laterality", None)
                    if side is not None:
                        side = str(side).strip().upper()
                    else:
                        side = "UNKNOWN"

                    view = None
                    if hasattr(dcm, "ViewCodeSequence"):
                        try:
                            seq = dcm.ViewCodeSequence
                            if len(seq) > 0:
                                item = seq[0]
                                view = getattr(item, "CodeMeaning", None) or getattr(item, "CodeValue", None)
                        except Exception:
                            view = None

                    if view is None:
                        view = getattr(dcm, "ViewPosition", None)

                    rows.append({
                        "path": path,
                        "patient_id": str(pid).strip(),
                        "side": side,  # "L", "R" ou "UNKNOWN"
                        "view": str(view).strip().upper() if view else "UNKNOWN",
                    })

                except Exception:
                    continue

    return pd.DataFrame(rows)


def merge_dataset(
    img_df: pd.DataFrame, labels_df: pd.DataFrame, out_path: str = "data/processed/dataset.csv"
) -> pd.DataFrame:
    """
    Merge DICOM images with clinical labels on `patient_id`.

    Keeps only patients that appear in both the DICOMs and the clinical file.

    Args:
        img_df: DataFrame from `extract_images` (columns: `path`, `patient_id`).
        labels_df: DataFrame from `load_labels` (columns: `patient_id`, `label`).
        out_path: Path where the merged CSV will be saved.

    Returns:
        Merged DataFrame and writes it to `out_path`.
    """
    # On ne garde que les images avec side L/R pour le merge
    img_df = img_df[img_df["side"].isin(["L", "R"])].copy()

    merged = img_df.merge(
        labels_df,
        on=["patient_id", "side"],
        how="inner"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("\nFINAL DATASET CREATED")
    print("Samples:", len(merged))
    print("Label distribution (image-level):")
    print(merged["label"].value_counts())

    return merged

def main(
    data_dir: str = "data",
    collection: str = "CMMD",
    max_cases: int = 20,
    force_download: bool = False,
):
    """
    End‑to‑end CMMD download + clinical + merge pipeline.

    Args:
        data_dir: Root directory for data (e.g., /data in Docker).
        collection: TCIA collection name.
        max_cases: Number of series to download (first `max_cases` in `SeriesInstanceUID`).
        force_download: If True, redownload images and clinical file even if cached.
    """
    raw_dir = os.path.join(data_dir, "raw", collection)
    processed_dir = os.path.join(data_dir, "processed")
    zips_dir = os.path.join(raw_dir, "zips")
    clinical_path = os.path.join(raw_dir, "clinical.xlsx")

    print("\n=== CMMD PIPELINE START ===")

    # 1. Get series metadata (cached)
    print("\n[1] Fetching metadata...")
    df = get_series(collection=collection, cache_dir=os.path.join(raw_dir))

    # 2. Download DICOM series
    print("[2] Downloading images...")
    for uid in tqdm(df["SeriesInstanceUID"].values[:max_cases]):
        try:
            download_series(uid, zips_dir, force=force_download)
        except Exception as e:
            print("Failed:", uid, e)

    # 3. Download and load clinical labels
    print("\n[3] Downloading clinical labels...")
    download_clinical(clinical_path, force=force_download)
    labels = load_labels(clinical_path)

    # 4. Extract DICOMs and get patient IDs
    print("\n[4] Extracting DICOM images...")
    dicom_dir = os.path.join(processed_dir, "dicom")
    img_df = extract_images(zips_dir, dicom_dir)

    # Only keep labels for patients whose images we downloaded
    labels = labels[labels["patient_id"].isin(img_df["patient_id"].unique())]

    # 5. Merge and save final dataset
    print("\n[5] Merging dataset...")
    dataset = merge_dataset(
        img_df,
        labels,
        out_path=os.path.join(processed_dir, "dataset.csv")
    )

    print("\n=== CMMD PIPELINE COMPLETE ===")
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser(
        description="CMMD Automated Download and Labeling Pipeline"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory (e.g., /data in Docker).",
    )
    parser.add_argument(
        "--collection",
        default="CMMD",
        help="TCIA collection name (default: CMMD).",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=20,
        help="Number of series to download (default: 20).",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Redownload images and clinical file even if they exist.",
    )

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        collection=args.collection,
        max_cases=args.max_cases,
        force_download=args.force_download,
    )
