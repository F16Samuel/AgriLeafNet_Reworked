# data/loader.py

from data.dataset import load_dataset
from data.transforms import get_augmentations, albumentations_preprocess_fn

def prepare_data(config):
    train_ds, val_ds, class_names = load_dataset(
        data_dir=config['data']['dataset_path'],
        image_size=tuple(config['data']['image_size']),
        batch_size=config['data']['batch_size'],
        val_split=config['data']['val_split'],
        seed=config['seed']
    )

    aug = get_augmentations(image_size=tuple(config['data']['image_size']))
    train_ds = train_ds.map(albumentations_preprocess_fn(aug))

    return train_ds, val_ds, class_names