# training/losses.py

import tensorflow as tf

def get_loss(name="categorical_crossentropy"):
    if name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    elif name == "label_smoothing":
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss: {name}")
