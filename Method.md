# 📚 Methodology: Leaf Disease Classification Pipeline using Augmented ResNet in TensorFlow

This section outlines the systematic methodology followed to design a robust image classification pipeline for plant leaf disease detection using a modified ResNet architecture, advanced augmentation strategies, explainability modules, and a modular, reproducible training workflow.

---

## 1. Dataset Preparation and Preprocessing

### 1.1 Data Sourcing

The project uses the **PlantVillage** dataset comprising RGB images of healthy and diseased leaves across 15 plant disease classes. Images are organized in a directory structure where each subdirectory corresponds to one class label.

### 1.2 Loading and Splitting

We leveraged `tf.keras.utils.image_dataset_from_directory()` to:

* Load the dataset with a fixed `seed` for reproducibility.
* Split the dataset into training and validation sets (using `validation_split` parameter).

### 1.3 Normalization and One-Hot Encoding

Each image is:

* Cast to `float32` and normalized to \[0, 1] range.
* Corresponding label is one-hot encoded based on the number of class names.

```python
image = tf.cast(image, tf.float32) / 255.0
label = tf.one_hot(label, depth=num_classes)
```

---

## 2. Advanced Data Augmentation with Albumentations

### 2.1 Motivation

To increase model generalization and tackle overfitting, we integrated **Albumentations**—a high-performance library for image augmentation.

### 2.2 Transformations Used

* **Training Mode Augmentations**:

  * `RandomResizedCrop`
  * `HorizontalFlip` and `VerticalFlip`
  * `HueSaturationValue`, `RandomBrightnessContrast` via `OneOf`
  * `Normalize` (ImageNet mean-std)

* **Validation Mode Augmentations**:

  * `Resize`
  * `Normalize`

### 2.3 TensorFlow Integration

Albumentations isn't TensorFlow-native, so a custom `albumentations_preprocess_fn()` was written to wrap its transformations inside `tf.numpy_function`, converting tensors to NumPy, applying augmentation, and converting them back.

```python
image = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
image.set_shape([224, 224, 3])
```

### 2.4 Pipeline Integration

To preserve the TensorFlow `tf.data.Dataset` pipeline’s efficiency:

* The dataset is **unbatched**, augmented, then **rebatch**ed.
* Augmented datasets are **prefetched** using `AUTOTUNE`.

---

## 3. Training Configuration and Hyperparameters

Configuration is centralized in a **YAML-based config file** (`train_config.yaml`) for reproducibility.

Key parameters:

* `batch_size`: 32
* `optimizer`: AdamW
* `learning_rate`: 0.0003
* `weight_decay`: 1e-4
* `epochs`: 100
* `scheduler`: Cosine Annealing with Warm Restarts
* `early_stopping_patience`: 10

These are dynamically parsed and passed to each module (dataset, trainer, model, callbacks).

---

## 4. Model Architecture and Modifications

### 4.1 Backbone

A **ResNet50** model (pretrained on ImageNet) was chosen for its feature extraction power and was further adapted with:

* Optional **LoRA-style low-rank adapters** to inject trainable components with fewer parameters.
* Custom **Skip Connections** and bottleneck tweaks (if used).
* Final dense layers matching `num_classes` with softmax activation.

### 4.2 Parameter Efficiency

* \~32M total params
* \~9M trainable (frozen base, trainable head & LoRA layers)
* Float32 weights optimized using mixed precision if GPU allows.

---

## 5. Handling Infinite Training and Dataset Repeats

**Issue**: Unbatched datasets with `.repeat()` caused infinite looping.
**Solution**: We manually computed `steps_per_epoch` as:

```python
steps_per_epoch = total_train_samples // batch_size
```

This ensured:

* Compatibility with `.repeat()`
* Proper dataset termination
* Safe use of `EarlyStopping` and `ReduceLROnPlateau` callbacks

---

## 6. Optimizer Choice and Behavior

**AdamW Optimizer** was chosen over classic Adam due to:

* Improved regularization via decoupled weight decay
* Adaptive learning rate for each parameter
* Bias correction and stability

The optimizer was configured using TensorFlow’s `tf.keras.optimizers.AdamW`.

---

## 7. Loss Monitoring and Interpretation

* The primary loss used: **Categorical Crossentropy**
* Observed training behavior:

  * Initial loss ≈ 3.3 (uniform prediction over 15 classes)
  * Gradual decrease with increasing accuracy
  * **Val loss spikes** were analyzed to detect label leaks or overfitting
  * Catastrophic val loss (e.g., `val_loss: 1934`) was tied to mismatches in label encoding and augmentation timing

---

## 8. Repeat Handling and Infinite Epoch Fix

To support `.repeat()` without infinite epochs:

* Dataset cardinality was extracted using `.cardinality().numpy()`
* Steps per epoch were explicitly passed to `model.fit(...)`

This ensured deterministic training duration.

---

## 9. Explainability Modules

Planned integrations (partially implemented):

* **Grad-CAM** on `conv5_block3_out` for saliency visualizations
* **LIME** with 1000 samples per image to explain predictions
* **t-SNE** visualizations of final-layer activations for class separability diagnostics

---

## 10. Logging, Callbacks & Monitoring

Implemented callbacks:

* `ModelCheckpoint`
* `EarlyStopping`
* `TensorBoard`
* `ReduceLROnPlateau`

All paths for logs and weights were configured from the `config` file, ensuring easy portability and resume capability.

---

## 11. Bugs and Iterative Fixes

We iteratively resolved key issues:

* Prefetch RAM budget exceeded → lowered batch size
* TensorFlow’s `OUT_OF_RANGE` error → added `.repeat()` with `steps_per_epoch`
* Label mismatch in augmentation flow → moved one-hot encoding post-augmentation
* Memory pressure during augmentation → optimized NumPy function wrapping

---

## ✅ Final Workflow Summary

```text
[TF Dataset Loader] → [Unbatch → Albumentations Augmentations → Rebatch] → [Prefetch]
                ↓
        [Custom Model (ResNet+LoRA)] ← [Config-driven Training Loop]
                ↓
   [Logging, Callbacks, GradCAM, LIME, t-SNE Outputs]
```