# data/transforms.py

import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.tensorflow import ToTensorV2
import tensorflow as tf
import numpy as np

def get_augmentations(image_size=(224, 224)):
    return A.Compose([
        A.RandomResizedCrop(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ], p=0.3),
        A.Normalize(),
    ])

def albumentations_preprocess_fn(augmentation):
    def wrap(image, label):
        def aug_fn(img):
            img = img.numpy()
            augmented = augmentation(image=img)["image"]
            return augmented

        image = tf.py_function(func=aug_fn, inp=[image], Tout=tf.float32)
        image.set_shape([224, 224, 3])
        return image, label
    return wrap
