# main.py

import argparse
import yaml
from training.train import train
from training.evaluate import evaluate_model
from explainability.grad_cam import save_gradcam
from explainability.lime_wrapper import get_lime_explanation
from explainability.tsne_plotter import extract_features, plot_tsne
from data.loader import prepare_data
import tensorflow as tf
import numpy as np
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="AgriLeafNet CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "explain"], help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--image_path", type=str, help="Path to input image for explanation")
    parser.add_argument("--explain_type", type=str, choices=["gradcam", "lime", "tsne"], help="Type of explanation")
    return parser.parse_args()

def load_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.expand_dims(tf.cast(image_rgb, tf.float32) / 255.0, axis=0)
    return image_tensor, image_rgb

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        train(args.config)

    elif args.mode == "test":
        test_ds, _, class_names = prepare_data(config)
        evaluate_model(args.model_path, test_ds, class_names)

    elif args.mode == "explain":
        model = tf.keras.models.load_model(args.model_path)
        image_tensor, original_image = load_image(args.image_path, tuple(config["data"]["image_size"]))
        class_names = list(range(config["data"]["num_classes"]))

        if args.explain_type == "gradcam":
            layer = config["explainability"]["grad_cam_layers"][0]
            save_gradcam(model, image_tensor, original_image, layer_name=layer,
                         save_path="outputs/visualizations/gradcam.png")

        elif args.explain_type == "lime":
            get_lime_explanation(model, original_image, class_names, save_path="outputs/visualizations/lime.png")

        elif args.explain_type == "tsne":
            train_ds, _, class_names = prepare_data(config)
            feats, labels = extract_features(model, train_ds)
            plot_tsne(feats, labels, class_names)

    elif args.mode == "predict":
        image_tensor, _ = load_image(args.image_path, tuple(config["data"]["image_size"]))
        model = tf.keras.models.load_model(args.model_path)
        preds = model.predict(image_tensor)[0]
        class_index = np.argmax(preds)
        confidence = preds[class_index] * 100
        print(f"🌱 Predicted Class Index: {class_index} — Confidence: {confidence:.2f}%")
        
if __name__ == "__main__":
    main()
