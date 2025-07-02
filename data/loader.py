# data/loader.py

from data.dataset import load_dataset
from data.transforms import get_augmentations, albumentations_preprocess_fn
import tensorflow as tf

def prepare_data(config):
    # Load raw train/val datasets and class labels
    train_ds, val_ds, class_names = load_dataset(
        data_dir=config['data']['dataset_path'],
        image_size=tuple(config['data']['image_size']),
        batch_size=config['data']['batch_size'],
        val_split=config['data']['val_split'],
        seed=config['seed']
    )

    # Apply Albumentations augmentations to training dataset
    aug = get_augmentations(image_size=tuple(config['data']['image_size']), is_train=True)
    
    # ⚠️ Unbatch → Augment → Rebatch (important)
    train_ds = train_ds.unbatch()
    train_ds = train_ds.map(
        albumentations_preprocess_fn(aug),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.repeat(1500)
    train_ds = train_ds.batch(config['data']['batch_size'])

    # Validation augmentations (if any)
    val_aug = get_augmentations(image_size=tuple(config['data']['image_size']), is_train=False)
    val_ds = val_ds.unbatch()
    val_ds = val_ds.map(
        albumentations_preprocess_fn(val_aug),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(config['data']['batch_size'])

    # Add prefetching
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
