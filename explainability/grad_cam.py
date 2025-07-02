# explainability/grad_cam.py

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def get_grad_cam(model, image_tensor, layer_name, class_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap, int(class_index)

def overlay_grad_cam(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = heatmap_color * alpha + image
    return np.clip(overlay, 0, 255).astype(np.uint8)

def save_gradcam(model, image_tensor, original_image, layer_name, save_path):
    heatmap, class_index = get_grad_cam(model, image_tensor, layer_name)
    cam = overlay_grad_cam(original_image, heatmap)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
    print(f"[✓] Grad-CAM saved at: {save_path}")
