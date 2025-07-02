# training/train.py

import os
import yaml
import tensorflow as tf
from models.agrileafnet import build_agrileafnet
from data.loader import prepare_data
from training.scheduler import CosineAnnealingScheduler
from training.early_stopping import CustomEarlyStopping
from training.losses import get_loss
from utils.seed import set_global_seed
from utils.logger import get_logger

def train(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_global_seed(config["seed"])
    logger = get_logger()

    train_ds, val_ds, class_names = prepare_data(config)
    model = build_agrileafnet(input_shape=tuple(config["data"]["image_size"]) + [3], num_classes=len(class_names))

    loss_fn = get_loss()
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.summary()

    callbacks = []

    if config["callbacks"]["early_stopping"]:
        callbacks.append(CustomEarlyStopping(patience=config["training"]["early_stopping_patience"]))

    if config["callbacks"]["tensorboard"]:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=config["training"]["log_dir"]))

    if config["callbacks"]["model_checkpoint"]:
        os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["training"]["checkpoint_dir"], "agrileafnet_best.keras"),
            monitor="val_accuracy", save_best_only=True
        ))

    if config["training"]["scheduler"] == "cosine_annealing":
        callbacks.append(CosineAnnealingScheduler(
            initial_lr=config["training"]["learning_rate"],
            t_max=10,
            eta_min=1e-6,
            restart_mult=2
        ))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks
    )

    return model, history
