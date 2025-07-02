# explainability/tsne_plotter.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import tensorflow as tf

def extract_features(model, dataset):
    feature_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # Assumes last dense layer before classifier
    )
    features, labels = [], []
    for imgs, lbls in dataset:
        feats = feature_model.predict(imgs)
        features.append(feats)
        labels.append(tf.argmax(lbls, axis=1).numpy())
    return np.concatenate(features), np.concatenate(labels)

def plot_tsne(features, labels, class_names, save_path="outputs/visualizations/tsne.png"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", legend="full")
    plt.title("t-SNE Feature Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] t-SNE plot saved at: {save_path}")
