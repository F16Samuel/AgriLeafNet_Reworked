# data/dataset.py

import tensorflow as tf
import os
from glob import glob

def load_dataset(data_dir, image_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.one_hot(label, depth=len(class_names))

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names
