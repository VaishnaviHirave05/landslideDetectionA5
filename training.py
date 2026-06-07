"""
training.py – Landslide Detection Model Training
=================================================
Trains a MobileNetV2-based binary classifier (landslide vs nonlandslide)
using the SAME architecture and preprocessing that Google Teachable Machine
exports, so the resulting landslide_model.h5 is fully compatible with the
provided prediction code.

Dataset folder structure expected:
    d:/landslide/
        landslide/          ← class 0
        Nonlandslide/       ← class 1

Output files:
    landslide_model.h5      ← saved Keras model
    labels.txt              ← class labels (Teachable Machine format)
    training_history.png    ← accuracy / loss curves
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress verbose TF logs
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from PIL import Image, ImageOps

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these if needed)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = r"D:\landslide"          # root of the project
CLASS_DIRS    = {                         # folder_name : label_index
    "landslide":    0,
    "Nonlandslide": 1,
}
IMG_SIZE      = (224, 224)               # Teachable Machine default
BATCH_SIZE    = 32
EPOCHS        = 30                       # fine-tuning epochs (early-stop applies)
FINE_TUNE_AT  = 100                      # unfreeze MobileNetV2 layers after epoch
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.3
TEST_SPLIT    = 0.2                      # 20 % held out for validation/testing
RANDOM_SEED   = 42

MODEL_OUT     = os.path.join(BASE_DIR, "landslide_model.h5")
LABELS_OUT    = os.path.join(BASE_DIR, "labels.txt")
HISTORY_OUT   = os.path.join(BASE_DIR, "training_history.png")

np.set_printoptions(suppress=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – Load & preprocess images
# ─────────────────────────────────────────────────────────────────────────────
def load_images(base_dir, class_dirs, img_size):
    """
    Loads every image from each class folder, resizes to img_size (centre-crop
    via ImageOps.fit, identical to the prediction code), then normalises to
    [-1, 1]  →  (pixel / 127.5) - 1  (Teachable Machine standard).
    Returns numpy arrays X (float32) and y (int).
    """
    X, y = [], []
    class_counts = {}

    for folder_name, label_idx in class_dirs.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}  – skipping.")
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        print(f"  → '{folder_name}'  ({label_idx}): {len(files)} images found")
        class_counts[folder_name] = 0

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                # same resize/crop as the prediction script
                img = ImageOps.fit(img, img_size, Image.Resampling.LANCZOS)
                arr = np.asarray(img, dtype=np.float32)
                # Teachable Machine normalisation: [-1, 1]
                arr = (arr / 127.5) - 1.0
                X.append(arr)
                y.append(label_idx)
                class_counts[folder_name] += 1
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")

    print(f"\nTotal images loaded: {len(X)}")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – Build model  (MobileNetV2 backbone + custom head)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes, img_size, dropout_rate):
    """
    Replicates the Teachable Machine export:
        MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128) → Dropout
        → Dense(num_classes, softmax)
    """
    base = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False                # freeze backbone initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    if num_classes == 2:
        # binary-style softmax (Teachable Machine uses softmax even for 2 classes)
        outputs = Dense(num_classes, activation="softmax")(x)
    else:
        outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – Plot training history
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Landslide Detection – Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",  color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",    color="#FF5722", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",  color="#4CAF50", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",    color="#9C27B0", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining history plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix – Test Set", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = save_path.replace(".png", "_cm.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved      → {cm_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Landslide Detection – Model Training")
    print("=" * 60)

    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nGPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\nNo GPU detected – training on CPU (may be slow).")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading images …")
    X, y = load_images(BASE_DIR, CLASS_DIRS, IMG_SIZE)

    if len(X) == 0:
        print("\n[ERROR] No images were loaded. Check your folder paths.")
        sys.exit(1)

    num_classes = len(CLASS_DIRS)
    class_names = sorted(CLASS_DIRS.keys(), key=lambda k: CLASS_DIRS[k])

    # ── Train / validation / test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,       # ~15 % of remaining = ~12 % of total
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    print(f"\n  Train : {len(X_train)} images")
    print(f"  Val   : {len(X_val)}   images")
    print(f"  Test  : {len(X_test)}  images")

    # one-hot encode labels
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[2/5] Building MobileNetV2 model …")
    model, base_model = build_model(num_classes, IMG_SIZE, DROPOUT_RATE)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Phase 1: train the custom head (backbone frozen) ──────────────────────
    print("\n[3/5] Phase 1 – Training classification head (backbone frozen) …")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: fine-tune top layers of MobileNetV2 ─────────────────────────
    print("\n[4/5] Phase 2 – Fine-tuning top MobileNetV2 layers …")
    base_model.trainable = True
    # only unfreeze layers after FINE_TUNE_AT
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # lower LR for fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_ft = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=15,                   # additional fine-tuning epochs
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # merge histories for plotting
    for key in history.history:
        history.history[key].extend(history_ft.history.get(key, []))

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"\n  Test Accuracy : {test_acc * 100:.2f} %")
    print(f"  Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test, y_pred, class_names, HISTORY_OUT)

    # ── Save model & labels ───────────────────────────────────────────────────
    model.save(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    # Write labels.txt in EXACT Teachable Machine format: "0 classname\n"
    with open(LABELS_OUT, "w") as f:
        for cls_name in class_names:
            idx = CLASS_DIRS[cls_name]
            f.write(f"{idx} {cls_name}\n")
    print(f"Labels saved → {LABELS_OUT}")

    # ── Plot history ──────────────────────────────────────────────────────────
    plot_history(history, HISTORY_OUT)

    print("\n" + "=" * 60)
    print("  Training complete! ✓")
    print(f"  Model   : {MODEL_OUT}")
    print(f"  Labels  : {LABELS_OUT}")
    print(f"  History : {HISTORY_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
