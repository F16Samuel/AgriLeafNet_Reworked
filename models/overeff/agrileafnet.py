# models/agrileafnet.py

import tensorflow as tf
from tensorflow.keras import layers, models
from models.custom_layers import FeatureHead, ResidualBlock
from models.lora_utils import LoRAAdapter

def build_agrileafnet(input_shape=(224, 224, 3), num_classes=15, lora_rank=8):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape, pooling=None
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)  # shape: (batch, 7, 7, 3072)

    skip_input = x  # Save for final skip connection

    '''
    # Stage 0
    fh1 = FeatureHead(2048, 2048, name="feature_head_0")(x)
    x = layers.Concatenate()([x, fh1])
    x = ResidualBlock(2048, name="res_block_0")(x)
    '''
    
    # Stage 1
    fh2 = FeatureHead(2048, 1024, name="feature_head_1")(x)
    x = layers.Concatenate()([x, fh2])
    x = ResidualBlock(2048, name="res_block_1")(x)

    # Stage 2
    fh2 = FeatureHead(2048, 512, name="feature_head_2")(x)
    x = layers.Concatenate()([x, fh2])
    x = ResidualBlock(2048, name="res_block_2")(x)

    # Stage 3
    fh3 = FeatureHead(2048, 2048, name="feature_head_3")(x)
    x = layers.Concatenate()([x, fh3])
    x = ResidualBlock(2048, name="res_block_3")(x)

    # Final skip connection to original ResNet output
    x = layers.Add()([x, skip_input])

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # LoRA adapter (dense-level)
    x = LoRAAdapter(units=2048, rank=lora_rank)(x)

    # Final classifier
    x = layers.Dense(512, activation='sigmoid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="AgriLeafNet")
