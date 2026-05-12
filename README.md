# CMMD Mammography Classification Pipeline

Deep learning pipeline for breast cancer classification on the CMMD dataset using DICOM mammography images, TensorFlow, and Docker.

Created by **Karim Makki**.

---

## Overview

This project provides an end-to-end pipeline for:

- Downloading CMMD mammography data from TCIA
- Extracting and merging clinical labels
- Building a clean dataset
- Performing patient-level train/validation/test splits
- Converting DICOM mammograms to PNG
- Mammography-specific preprocessing
- Training a patch-based attention model using DenseNet121

The pipeline is fully Dockerized and reproducible.

---

## Features

- Automated TCIA download pipeline
- Clinical label integration
- Patient-level stratified splitting
- Mammography-oriented preprocessing
- DICOM → PNG conversion
- Parallel CPU preprocessing
- Patch extraction with attention pooling
- DenseNet121 transfer learning
- Focal loss for class imbalance
- Oversampling + augmentation
- Breast-level and image-level evaluation
- Docker-based workflow

---

## Dataset

Dataset used:

- CMMD (Chinese Mammography Database)
- Source: The Cancer Imaging Archive (TCIA)

Clinical labels are automatically downloaded and merged during preprocessing.

---

## Project Structure

```text
project/
│
├── Dockerfile
├── Dockerfile.train
├── docker-compose.yml
├── requirements.txt
├── requirements.dev.txt
│
├── src/
│   ├── download/
│   │   └── cmmd_api_download.py
│   │
│   ├── preprocess/
│   │   ├── build_dataset.py
│   │   ├── split_dataset.py
│   │   └── prepare_images.py
│   │
│   └── train_mammo.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── models/
```

---

## Pipeline

### 1. Download and Prepare Dataset

The following pipeline:

- downloads CMMD DICOM series,
- downloads clinical labels,
- extracts DICOM files,
- merges labels,
- cleans the dataset,
- performs patient-level split,
- converts images to PNG.

Run:

```bash
docker compose run \
  --volume "$(pwd)":/app \
  --volume "$(pwd)/data":/data \
  --entrypoint=sh \
  cmmd-download \
  -c '
    python -m src.download.cmmd_api_download --data_dir /data --max_cases 1872 && \
    python -m src.preprocess.build_dataset && \
    python -m src.preprocess.split_dataset && \
    python -m src.preprocess.prepare_images
  '
```

---

### 2. Train the Model

Run:

```bash
sudo docker compose run --build cmmd-train
```

The trained model will be saved in:

```text
models/mammo_patch_attention.keras
```

---

## Preprocessing

The preprocessing pipeline includes:

- DICOM parsing
- Mammography filtering
- Breast orientation alignment
- Background artifact removal
- Intensity normalization
- CLAHE contrast enhancement
- Breast-centered cropping
- Resize with padding
- Patch extraction
- Parallel image conversion

---

## Model Architecture

The model uses:

- DenseNet121 pretrained on ImageNet
- Patch-based processing
- Attention pooling
- Two-stage training:
  - frozen backbone
  - fine-tuning

Training includes:

- focal loss
- oversampling
- data augmentation
- AUC monitoring
- learning rate scheduling
- early stopping

---

## Evaluation

Evaluation is performed at:

- image level
- breast level

Metrics include:

- Accuracy
- AUC
- Precision
- Recall
- F1-score
- Sensitivity
- Specificity

Threshold optimization is performed using Youden’s index on the validation set.

---

## Docker Services

### `cmmd-download`

Responsible for:

- dataset download
- preprocessing
- image preparation

### `cmmd-train`

Responsible for:

- TensorFlow training
- evaluation
- model export

---

## Dependencies

Main libraries:

- TensorFlow
- OpenCV
- pydicom
- pandas
- scikit-learn
- Pillow
- matplotlib

---

## Hardware Used

Training and preprocessing were developed on:

- Ubuntu 18.04 LTS
- Intel Core i5-6200U
- 12 GB RAM
- Intel HD Graphics 520
- AMD Radeon R5 M330

The preprocessing pipeline is CPU-oriented and optimized for low-resource environments.

---

## Notes

- The pipeline is resumable and idempotent.
- Already downloaded series are skipped automatically.
- Patient-level splitting prevents data leakage.
- PNG conversion only keeps valid mammography views.

---

## License

This project is intended for research and educational purposes.

Dataset usage remains subject to TCIA/CMMD licensing terms.
