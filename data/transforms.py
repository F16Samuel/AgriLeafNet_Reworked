# data/transforms.py

import albumentations as A
from albumentations.core.composition import OneOf
import tensorflow as tf
import numpy as np

def get_augmentations(image_size=(224, 224), is_train=True):
    """
    Returns an Albumentations Compose object based on mode.
    """
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Resize(*image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

def albumentations_preprocess_fn(augmentation):
    """
    Returns a tf.data-compatible wrapper that applies Albumentations to each image.
    """
    def wrap(image, label):
        def aug_fn(img):
            augmented = augmentation(image=img)["image"]
            return augmented.astype(np.float32)

        image = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
        image.set_shape([224, 224, 3])
        return image, label
    return wrap
