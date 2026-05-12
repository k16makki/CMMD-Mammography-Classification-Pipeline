"""
CMMD Image Preprocessing Step (Parallel, CPU‑only)

Converts DICOM files to PNG images in parallel without using GPU.
Uses threads for I/O and CPU‑bound image operations.

Input:
    data/processed/dataset_split.csv   (with paths like /data/processed/dicom/...)

Output:
    data/processed/png/
    data/processed/dataset_split_png.csv
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_mammography_orientation_and_keep(ds) -> tuple[bool, bool, bool]:
    """Minimal orientation filter for classification, aligning all breasts to the right."""
    modality = ds.get("Modality", "")
    laterality = ds.get("Laterality", "")
    image_laterality = ds.get("ImageLaterality", "")

    # Only keep mammography images
    if modality != "MG":
        return False, False, False

    lat = laterality if laterality in ("L", "R") else image_laterality
    if lat not in ("L", "R"):
        return False, False, False

    # Decide whether to flip so breast ends up on the RIGHT side:
    # - L‑breast: flip → moves from left to right
    # - R‑breast: no flip → already on right
    flip_horizontal = (lat == "L")

    is_left_breast = lat == "L"
    keep_for_training = True

    return keep_for_training, flip_horizontal, is_left_breast


def get_breast_mask(img: np.ndarray) -> np.ndarray:
    """
    Get a rough breast mask using Otsu‑like thresholding.
    Assumes background is roughly 0 and the breast is brighter.
    """
    img_uint8 = img.astype(np.uint8)
    # Global threshold (Otsu‑style)
    threshold = max(1, int(np.percentile(img_uint8, 5)))  # conservative low threshold
    mask = (img_uint8 > threshold).astype(np.uint8)

    # Optional: keep only the largest connected component
    try:
        import cv2
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if nlabels > 1:
            # Find largest component excluding background (label 0)
            areas = stats[1:, -1]
            idx = np.argmax(areas) + 1
            mask = (labels == idx).astype(np.uint8)
    except ImportError:
        pass  # keep simple mask

    return mask


def dicom_to_png(
    dicom_path: str, png_path: str, size: Tuple[int, int] = (224, 224)
) -> Optional[str]:
    """
    Convert a DICOM file to a normalized PNG image with robust preprocessing.

    Args:
        dicom_path: Path to the input DICOM file.
        png_path: Path where the PNG will be saved.
        size: Target image size.

    Returns:
        png_path on success, None if should not be kept.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)

        if img.ndim > 2:
            img = img[..., 0]

        keep, flip_horizontal, _ = get_mammography_orientation_and_keep(ds)
        if not keep:
            return None

        # --- 1. Truncate (percentile‑based) ---
        # Use breast mask to ignore background
        mask = get_breast_mask(img)
        region = img[mask > 0]
        if len(region) == 0:
            return None

        # Use 1–99% range within breast
        low = np.percentile(region, 1.0)
        high = np.percentile(region, 99.0)
        img = np.clip(img, low, high)

        # --- 2. Normalize to [0, 1] ---
        if high > low:
            img = (img - low) / (high - low)
        else:
            img = np.zeros_like(img)

        # --- 3. Optional: CLAHE (mild contrast enhancement) ---
        # This is optional; you can comment out if you prefer raw normalized
        # try:
        #     import cv2
        #     img_uint8 = (img * 255.0).astype(np.uint8)
        #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #     img_uint8 = clahe.apply(img_uint8)
        #     img = img_uint8.astype(np.float32) / 255.0
        # except ImportError:
        #     pass  # keep normalized image

        # --- 4. Flip to align breast to left (optional) ---
        image = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
        if flip_horizontal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # --- 5. Resize to fixed size ---
        image = image.resize(size, Image.Resampling.LANCZOS)

        # --- 6. Save ---
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        image.save(png_path)

        return png_path

    except Exception as e:
        print(f"=== Exception in dicom_to_png for {dicom_path}: {e}")
        return None


def prepare_images(
    input_csv: str = "data/processed/dataset_split.csv",
    output_csv: str = "data/processed/dataset_split_png.csv",
    output_dir: str = "data/processed/png",
    size: Tuple[int, int] = (224, 224),
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Convert DICOM images to PNG with robust preprocessing for classification.

    Args:
        input_csv: Input CSV with columns: path, patient_id, label, split.
        output_csv: CSV to save with PNG paths.
        output_dir: Folder where PNG images will be written (under /app).
        size: Target image size for all outputs.
        max_workers: Number of parallel threads.

    Returns:
        DataFrame with new paths (PNG) and updated metadata.
    """
    df = pd.read_csv(input_csv)
    print("=== CMMD PREPARE IMAGES (PARALLEL, CPU‑ONLY) ===")
    print("Input dataset shape (before orientation filter):", df.shape)

    # Ensure the root PNG folder exists under /data (Docker mount point)
    # inside the container, this will be /data/processed/png
    # We still write the CSV paths with /data instead of /app
    os.makedirs(os.path.join("/data", output_dir), exist_ok=True)

    rows = []
    futures = []
    failed_count = 0
    filtered_out_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            dicom_path = row["path"]  # e.g., /data/processed/dicom/<UID>/<slice>.dcm

            # Extract UID and slice from DICOM path
            rel = os.path.relpath(dicom_path, "/data/processed/dicom")
            uid = os.path.dirname(rel)
            bn = os.path.basename(rel)          # 00000001.dcm
            name = os.path.splitext(bn)[0]      # 00000001

            # Flatten all PNGs: <UID>_<slice>.png in the root PNG folder
            png_name = f"{uid}_{name}.png"
            # Write PNG to /data/processed/png/... directly (inside container)
            png_path = os.path.join("/data", output_dir, png_name)

            if os.path.exists(png_path):
                # Already exists → keep row and reuse PNG path
                row = row.copy()
                row["path"] = png_path
                rows.append(row)
            else:
                # Submit for conversion
                # Also change where we save PNG on the host: /data/processed/png
                # On the host, this is ~/Desktop/Hera-mi_project/data/processed/png
                future = executor.submit(dicom_to_png, dicom_path, png_path, size)
                futures.append((future, _, row))

        # Collect results
        for future, idx, row in futures:
            result = future.result()
            if result is not None:
                row = row.copy()
                row["path"] = result
                rows.append(row)
            else:
                failed_count += 1

    # Build new dataframe (only images that passed the filter)
    df_new = pd.DataFrame(rows)

    # Ensure CSV directory exists (still under /app in container, but mounted)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_new.to_csv(output_csv, index=False)

    print("DICOMs filtered out by orientation / modality:", filtered_out_count)
    print("Failed DICOMs (no PNG generated):", failed_count)
    print("PNG dataset saved to:", output_csv)
    print("PNG images saved in:", os.path.join("/data", output_dir))
    print("Final dataset shape (only training‑ready views):", df_new.shape)

    return df_new


if __name__ == "__main__":
    prepare_images()
