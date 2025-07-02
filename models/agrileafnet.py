# models/agrileafnet.py

import tensorflow as tf
from tensorflow.keras import layers, models
from models.custom_layers import FeatureHead, ResidualBlock
from models.lora_utils import LoRAAdapter

def build_agrileafnet(input_shape=(224, 224, 3), num_classes=15, lora_rank=4):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling='avg'
    )
    base_model.trainable = False  # For initial transfer learning phase

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)

    # Add custom residual + LoRA + feature heads
    x = ResidualBlock(2048)(x)
    x = LoRAAdapter(units=2048, rank=lora_rank)(x)
    x = FeatureHead(units=256, name="feature_head_256")(x)
    x = FeatureHead(units=128, name="feature_head_128")(x)


    # Skip Connection
    skip = layers.Dense(128)(x)
    x = layers.Concatenate()([x, skip])

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="AgriLeafNet")
