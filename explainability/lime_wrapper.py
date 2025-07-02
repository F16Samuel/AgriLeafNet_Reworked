# explainability/lime_wrapper.py

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os

def get_lime_explanation(model, image_np, class_names, save_path="outputs/visualizations/lime.png"):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        return model.predict(images)

    explanation = explainer.explain_instance(
        image_np.astype('double'),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    output_img = mark_boundaries(temp / 255.0, mask)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.imsave(save_path, output_img)
    print(f"[✓] LIME explanation saved at: {save_path}")
