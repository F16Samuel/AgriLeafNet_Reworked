# tests/test_dataset.py

from data.dataset import load_dataset

def test_data_loader_shapes():
    train_ds, val_ds, class_names = load_dataset("data/PlantVillage", image_size=(224, 224), batch_size=8)
    for images, labels in train_ds.take(1):
        assert images.shape[1:] == (224, 224, 3)
        assert labels.shape[1] == len(class_names)