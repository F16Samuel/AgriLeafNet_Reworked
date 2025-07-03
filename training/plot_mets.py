# training/plot_mets.py

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint


class StepwiseMetricsLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"[Train Step] {batch + 1} | Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"[Val Step] {batch + 1} | Val Loss: {logs['loss']:.4f}, Val Accuracy: {logs['accuracy']:.4f}")


def plot_metrics(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)

    # Plot: Loss
    plt.figure()
    plt.plot(epochs, metrics['loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Plot: Accuracy
    plt.figure()
    plt.plot(epochs, metrics['accuracy'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()