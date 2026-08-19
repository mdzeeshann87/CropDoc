"""
Optional: train a real CNN and drop it into models/crop_disease_model.h5.

Once that file exists, inference.py automatically switches from the
rule-based color/texture analyzer to this trained model — no other code
changes needed.

Usage:
    pip install tensorflow
    python train_model.py --data_dir /path/to/dataset --crop tomato --epochs 15

Expected data_dir layout (standard Keras "flow_from_directory" format),
one folder per class, e.g. for the PlantVillage dataset:

    data_dir/
      Tomato___healthy/
      Tomato___Early_blight/
      Tomato___Late_blight/
      Tomato___Leaf_Mold/
      ...

Class folder names should be mapped to the `id` fields in disease_data.py's
condition list for the crop you're training, in the same order, so
predictions line up with the app's labels (see CLASS_ORDER below — edit it
to match your dataset's folder names before training).
"""

import argparse
import os

# Folder-name -> condition id mapping. Edit to match your dataset.
CLASS_ORDER = {
    "tomato": ["tomato_healthy", "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold"],
}


def build_model(num_classes, img_size=224):
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # start with a frozen backbone; unfreeze later for fine-tuning

    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1.0 / 255),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to labeled leaf-image dataset")
    parser.add_argument("--crop", required=True, choices=CLASS_ORDER.keys())
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default=os.path.join("models", "crop_disease_model.h5"))
    args = parser.parse_args()

    from tensorflow import keras

    train_ds = keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=(args.img_size, args.img_size), batch_size=args.batch_size, label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=(args.img_size, args.img_size), batch_size=args.batch_size, label_mode="categorical",
    )

    print("Detected classes (must match CLASS_ORDER order):", train_ds.class_names)

    num_classes = len(train_ds.class_names)
    model = build_model(num_classes, args.img_size)

    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()
