import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ======================
# CONFIG
# ======================

IMG_SIZE = 160
PATCH_SIZE = 64

TOP_K_PATCHES = 4

BATCH_SIZE = 4

AUTOTUNE = tf.data.AUTOTUNE

MEAN_GLOBAL = None
STD_GLOBAL = None

DATA_ROOT = os.environ.get("DATA_ROOT", "/app")


# ======================
# PATH
# ======================
def fix_path(path):

    path = str(path)

    if os.path.exists(path):
        return path

    path = path.lstrip("/")

    candidate = os.path.join(DATA_ROOT, path)

    if os.path.exists(candidate):
        return candidate

    return candidate


# ======================
# PREPROCESS
# ======================
def remove_border_artifacts(img):

    img = img.copy()

    img[:5, :] = 0
    img[-5:, :] = 0
    img[:, :5] = 0
    img[:, -5:] = 0

    return img


def normalize_image(img):

    p1, p99 = np.percentile(img, (1, 99))

    img = np.clip(img, p1, p99)

    return (img - p1) / (p99 - p1 + 1e-6)


def apply_clahe(img):

    img = normalize_image(img)

    img_u8 = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    out = clahe.apply(img_u8)

    return out.astype(np.float32) / 255.0


def crop_breast(gray):

    gray = gray.copy()

    gray = remove_border_artifacts(gray)

    norm = normalize_image(gray)

    coords = np.argwhere(norm > 0.05)

    if len(coords) < 50:

        mask = np.ones_like(gray, dtype=np.uint8) * 255

        return gray, mask

    cy, cx = coords.mean(axis=0).astype(int)

    crop_h, crop_w = 512, 512

    y0 = max(0, cy - crop_h // 2)
    x0 = max(0, cx - crop_w // 2)

    y1 = min(gray.shape[0], y0 + crop_h)
    x1 = min(gray.shape[1], x0 + crop_w)

    cropped = gray[y0:y1, x0:x1]

    pad_y = crop_h - cropped.shape[0]
    pad_x = crop_w - cropped.shape[1]

    cropped = cv2.copyMakeBorder(
        cropped,
        0,
        pad_y,
        0,
        pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    mask = np.zeros_like(gray, dtype=np.uint8)

    mask[y0:y1, x0:x1] = 255

    return cropped, mask


def resize_with_padding(img, size=IMG_SIZE):

    coords = np.argwhere(img > 0.05)

    if len(coords) > 0:

        cy, cx = coords.mean(axis=0).astype(int)

        h, w = img.shape

        shift_y = h // 2 - cy
        shift_x = w // 2 - cx

        M = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderValue=0
        )

    h, w = img.shape

    scale = min(size / h, size / w)

    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_CUBIC
    )

    top = (size - nh) // 2
    bottom = size - nh - top

    left = (size - nw) // 2
    right = size - nw - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )


# ======================
# PATCH EXTRACTION
# ======================
def extract_topk_patches(img):

    h, w = img.shape

    stride = PATCH_SIZE // 2

    candidates = []

    for y in range(0, h - PATCH_SIZE + 1, stride):

        for x in range(0, w - PATCH_SIZE + 1, stride):

            patch = img[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            #score = patch.max() + patch.std()
            score = (
            0.7 * patch.mean()
            + 1.5 * patch.std()
            )

            candidates.append(
                (score, patch)
            )

    candidates.sort(
        key=lambda z: z[0],
        reverse=True
    )

    patches = []

    for i in range(TOP_K_PATCHES):

        if i < len(candidates):

            patch = candidates[i][1]

        else:

            patch = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32
            )

        patch = cv2.resize(
            patch,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        patch = np.stack(
            [patch, patch, patch],
            axis=-1
        )

        patches.append(patch)

    patches = np.array(
        patches,
        dtype=np.float32
    )

    return patches


# ======================
# DEBUG
# ======================
def preprocess_debug(path):

    path = fix_path(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:

        print("[WARN] image introuvable:", path)

        return

    img = img.astype(np.float32) / 255.0

    img = remove_border_artifacts(img)

    img = normalize_image(img)

    cropped, mask = crop_breast(img)

    cropped = normalize_image(cropped)

    clahe_img = apply_clahe(cropped)

    final_img = resize_with_padding(
        clahe_img,
        IMG_SIZE
    )

    patches = extract_topk_patches(
        final_img
    )

    fig, axes = plt.subplots(
        2,
        6,
        figsize=(18, 6)
    )

    items = [
        (img, "original"),
        (mask, "mask"),
        (cropped, "crop"),
        (clahe_img, "clahe"),
        (final_img, "final")
    ]

    for i, (im, title) in enumerate(items):

        axes[0, i].imshow(im, cmap="gray")
        axes[0, i].set_title(title)
        axes[0, i].axis("off")

    for k in range(TOP_K_PATCHES):

        axes[1, k].imshow(
            patches[k][:, :, 0],
            cmap="gray"
        )

        axes[1, k].set_title(f"patch {k+1}")
        axes[1, k].axis("off")

    plt.tight_layout()

    plt.show()


# ======================
# PREPROCESS IMAGE
# ======================
def preprocess_image(path):

    path = path.numpy().decode("utf-8")

    path = fix_path(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:

        patches = np.zeros(
            (
                TOP_K_PATCHES,
                IMG_SIZE,
                IMG_SIZE,
                3
            ),
            dtype=np.float32
        )

        return patches

    img = img.astype(np.float32) / 255.0

    img = remove_border_artifacts(img)

    img = normalize_image(img)

    cropped, _ = crop_breast(img)

    cropped = normalize_image(cropped)

    clahe_img = apply_clahe(cropped)

    final_img = resize_with_padding(
        clahe_img,
        IMG_SIZE
    )

    patches = extract_topk_patches(
        final_img
    )

    if MEAN_GLOBAL is not None and STD_GLOBAL is not None:

        patches = (
            patches - MEAN_GLOBAL
        ) / (STD_GLOBAL + 1e-6)

    return patches.astype(np.float32)


def load_image(path, label):

    patches = tf.py_function(
        preprocess_image,
        [path],
        Tout=tf.float32
    )

    patches.set_shape(
        (
            TOP_K_PATCHES,
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    return patches, tf.cast(label, tf.float32)


# ======================
# AUGMENTATION
# ======================
def augment_patches(patches):

    out = []

    for i in range(TOP_K_PATCHES):

        img = patches[i]

        #img = tf.image.random_flip_left_right(img)

        img = tf.image.random_brightness(
            img,
            0.08
        )

        img = tf.image.random_contrast(
            img,
            0.95,
            1.05
        )

        img = tf.clip_by_value(
            img,
            -5.0,
            5.0
        )

        out.append(img)

    return tf.stack(out)


def augment(patches, label):

    patches = augment_patches(patches)

    return patches, label


# ======================
# DATASET
# ======================
def make_dataset(
    df,
    training=True,
    augment_factor=1
):

    if not training:

        ds = tf.data.Dataset.from_tensor_slices(
            (
                df["path"].values,
                df["label"].values
            )
        )

        ds = ds.map(
            load_image,
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    # =========================
    # TRAINING DATASET
    # =========================

    minority_class = (
        df["label"]
        .value_counts()
        .idxmin()
    )

    majority_class = (
        df["label"]
        .value_counts()
        .idxmax()
    )

    df_min = df[
        df["label"] == minority_class
    ]

    df_maj = df[
        df["label"] == majority_class
    ]

    print("\n=== OVERSAMPLING ===")
    print("Minority class:", minority_class)
    print("Majority class:", majority_class)

    print(
        "Minority before:",
        len(df_min)
    )

    print(
        "Majority:",
        len(df_maj)
    )

    # -------------------------
    # DATASET MAJORITAIRE
    # -------------------------

    ds_maj = tf.data.Dataset.from_tensor_slices(
        (
            df_maj["path"].values,
            df_maj["label"].values
        )
    )

    ds_maj = ds_maj.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    # -------------------------
    # DATASET MINORITAIRE
    # -------------------------

    ds_min = tf.data.Dataset.from_tensor_slices(
        (
            df_min["path"].values,
            df_min["label"].values
        )
    )

    ds_min = ds_min.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    datasets_min = [ds_min]

    for _ in range(augment_factor - 1):

        aug_ds = ds_min.map(
            augment,
            num_parallel_calls=AUTOTUNE
        )

        datasets_min.append(aug_ds)

    ds_min_final = datasets_min[0]

    for extra_ds in datasets_min[1:]:

        ds_min_final = ds_min_final.concatenate(extra_ds)

    # -------------------------
    # CONCAT
    # -------------------------

    ds = ds_maj.concatenate(ds_min_final)

    effective_minority = (
        len(df_min) * augment_factor
    )

    print(
        "Minority after augmentation:",
        effective_minority
    )

    print(
        "Final dataset size:",
        len(df_maj) + effective_minority
    )

    ds = ds.shuffle(512)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds


# ======================
# STATS
# ======================
def compute_global_stats(df):

    n_pixels = 0
    sum_ = 0.0
    sum_sq = 0.0
    n_valid = 0

    for _, row in df.iterrows():

        path = fix_path(row["path"])

        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = img.astype(np.float32) / 255.0

        img = remove_border_artifacts(img)

        img = normalize_image(img)

        cropped, _ = crop_breast(img)

        cropped = normalize_image(cropped)

        clahe_img = apply_clahe(cropped)

        final_img = resize_with_padding(
            clahe_img,
            IMG_SIZE
        )

        pixels = final_img.ravel()

        n_pixels += pixels.size

        sum_ += pixels.sum()

        sum_sq += (pixels ** 2).sum()

        n_valid += 1

        if n_valid % 500 == 0:

            print(
                f"[STATS] Processed {n_valid} images..."
            )

    mean = sum_ / n_pixels

    std = np.sqrt(
        sum_sq / n_pixels - mean ** 2
    )

    return mean, std


def standardize_dataset_stats(
    mean_global,
    std_global
):

    global MEAN_GLOBAL, STD_GLOBAL

    MEAN_GLOBAL = mean_global
    STD_GLOBAL = std_global


# ======================
# LOSS
# ======================
def binary_focal_loss(
    alpha_for_positive=0.35,
    gamma=2.0
):

    def loss(y_true, y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        ce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true)
            * tf.math.log(1.0 - y_pred)
        )

        p_t = (
            y_true * y_pred
            + (1.0 - y_true)
            * (1.0 - y_pred)
        )

        alpha_t = (
            y_true * alpha_for_positive
            + (1.0 - y_true)
            * (1.0 - alpha_for_positive)
        )

        focal = (
            alpha_t
            * tf.pow(1.0 - p_t, gamma)
            * ce
        )

        return tf.reduce_mean(focal)

    return loss


# ======================
# MODEL
# ======================
def build_model():

    backbone = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    backbone.trainable = False

    inputs = layers.Input(
        shape=(
            TOP_K_PATCHES,
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    x = layers.TimeDistributed(
        backbone
    )(inputs)

    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(x)

    attention = layers.Dense(
        64,
        activation="tanh"
    )(x)

    attention = layers.Dense(
        1
    )(attention)

    attention = layers.Softmax(axis=1)(
        attention
    )

    x = layers.Multiply()([
        x,
        attention
    ])

    x = layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1)
    )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    return model, backbone

def unfreeze_backbone(
    backbone,
    n_last_layers=60
):

    backbone.trainable = True

    for layer in backbone.layers[:-n_last_layers]:

        layer.trainable = False


# ======================
# METRICS
# ======================
def compute_metrics(
    y_true,
    y_score,
    threshold=0.5,
    tag=""
):

    y_pred = (
        y_score >= threshold
    ).astype(int)

    acc = (y_pred == y_true).mean()

    auc = roc_auc_score(
        y_true,
        y_score
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    print(f"\n=== {tag} ===")

    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


def get_scores(model, ds):

    y_true = []
    y_score = []

    for x, y in ds:

        p = model.predict(
            x,
            verbose=0
        )

        y_true.extend(y.numpy())

        y_score.extend(
            p.reshape(-1)
        )

    return (
        np.array(y_true),
        np.array(y_score)
    )


# ======================
# BREAST LEVEL
# ======================
def aggregate_by_breast(
    df_split,
    y_score,
    split_name="test"
):

    df_sub = df_split[
        df_split["split"] == split_name
    ].copy()

    assert len(df_sub) == len(y_score)

    df_sub["score"] = y_score

    df_breast = (
        df_sub
        .groupby(
            ["patient_id", "side"],
            as_index=False
        )
        .agg({
            "label": "max",
            "score": "max"
        })
    )

    y_true_b = df_breast["label"].values

    y_score_b = df_breast["score"].values

    auc_b = roc_auc_score(
        y_true_b,
        y_score_b
    )

    print(
        f"\nAUC breast-level ({split_name}): "
        f"{auc_b:.4f}"
    )

    print(
        "Breast-level label distribution:",
        np.bincount(y_true_b.astype(int))
    )

    return y_true_b, y_score_b


# ======================
# MAIN
# ======================
def main(args):

    df = pd.read_csv(
        os.path.join(
            args.data_dir,
            "processed/dataset_split_png.csv"
        )
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print("\n=== DISTRIBUTION TRAIN RAW ===")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n=== DISTRIBUTION VAL ===")
    print(
        val_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n=== DISTRIBUTION TEST ===")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    class_weight = None
    #{
    #    0: 1.8,
    #    1: 1.0
    #}

    print("\n=== CLASS WEIGHTS ===")
    print(class_weight)

    print("\n=== DEBUG IMAGE ===")

    preprocess_debug(
        train_df["path"].iloc[0]
    )

    print(
        "\n=== COMPUTING GLOBAL STATS ON TRAIN ==="
    )

    mean_global, std_global = compute_global_stats(
        train_df
    )

    print("mean_global =", mean_global)

    print("std_global =", std_global)

    standardize_dataset_stats(
        mean_global,
        std_global
    )

    train_ds = make_dataset(
        train_df,
        training=True,
        augment_factor=2
    )

    val_ds = make_dataset(
        val_df,
        training=False
    )

    test_ds = make_dataset(
        test_df,
        training=False
    )

    model, backbone = build_model()

    print(
        "\n=== MODEL SUMMARY ==="
    )

    model.summary()

    print(
        "\n=== TRAIN STAGE 1 (backbone frozen) ==="
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            3e-4
        ),

        loss=binary_focal_loss(
            alpha_for_positive=0.35,
            gamma=2.0
        ),

        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    es = EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor="val_auc",
        mode="max"
    )

    lr = ReduceLROnPlateau(
        patience=2,
        factor=0.5,
        monitor="val_auc",
        mode="max"
    )

    history_stage1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=8,

        callbacks=[es, lr],

        class_weight=class_weight,

        verbose=1
    )

    print(
        "\n=== TRAIN STAGE 2 (fine-tuning backbone) ==="
    )

    unfreeze_backbone(
        backbone,
        n_last_layers=60
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            1e-5
        ),

        loss=binary_focal_loss(
            alpha_for_positive=0.3,
            gamma=2.0
        ),

        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    es_ft = EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor="val_auc",
        mode="max"
    )

    lr_ft = ReduceLROnPlateau(
        patience=2,
        factor=0.5,
        monitor="val_auc",
        mode="max"
    )

    history_stage2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=4,

        callbacks=[es_ft, lr_ft],

        class_weight=class_weight,

        verbose=1
    )

    print(
        "\n=== EVALUATION (thr=0.5 pour info) ==="
    )

    y_val, s_val = get_scores(
        model,
        val_ds
    )

    y_test, s_test = get_scores(
        model,
        test_ds
    )

    compute_metrics(
        y_val,
        s_val,
        0.5,
        "VAL_thr=0.5 (image)"
    )

    compute_metrics(
        y_test,
        s_test,
        0.5,
        "TEST_thr=0.5 (image)"
    )

    print(
        "\n=== BREAST-LEVEL AUC (before threshold search) ==="
    )

    y_val_b, s_val_b = aggregate_by_breast(
        df,
        s_val,
        "val"
    )

    y_test_b, s_test_b = aggregate_by_breast(
        df,
        s_test,
        "test"
    )

    best_thr = 0.5
    best_j = -1

    print(
        "\n=== THRESHOLD SEARCH ==="
    )

    for thr in np.linspace(0.05, 0.95, 19):

        y_pred_thr = (
            s_val_b >= thr
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val_b,
            y_pred_thr
        ).ravel()

        sens = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 0.0
        )

        spec = (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0.0
        )

        j = sens + spec - 1

        print(
            f"thr={thr:.2f} "
            f"sens={sens:.4f} "
            f"spec={spec:.4f} "
            f"J={j:.4f}"
        )

        if j > best_j:

            best_j = j
            best_thr = thr

    print(
        "\n=== BEST THRESHOLD ON VAL (BREAST-LEVEL, YOUDEN) ==="
    )

    print(
        "best_thr =",
        best_thr,
        "J =",
        best_j
    )

    print(
        "\n=== FINAL EVALUATION WITH BEST THR (breast-level) ==="
    )

    compute_metrics(
        y_val_b,
        s_val_b,
        best_thr,
        f"VAL_breast_thr={best_thr:.2f}"
    )

    compute_metrics(
        y_test_b,
        s_test_b,
        best_thr,
        f"TEST_breast_thr={best_thr:.2f}"
    )

    print(
        "\n=== SAVING MODEL ==="
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        "models/mammo_patch_attention.keras"
    )

    print(
        "\nModel saved to:"
    )

    print(
        "models/mammo_patch_attention.keras"
    )

    print(
        "\n=== FINISHED SUCCESSFULLY ==="
    )


# ======================
# ENTRYPOINT
# ======================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    args = parser.parse_args()

    print("\n=== STARTING TRAINING ===")

    print("TensorFlow version:", tf.__version__)

    print("Num GPUs Available:",
          len(tf.config.list_physical_devices('GPU')))

    main(args)

