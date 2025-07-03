import tensorflow as tf
from tensorflow.keras import layers, models
from models.custom_layers import FeatureHead, ResidualBlock
from models.lora_utils import LoRAAdapter

def build_agrileafnet(input_shape=(160, 160, 3), num_classes=15, lora_rank=8):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape, pooling=None
    )
    base_model.trainable = False  # Start with frozen backbone

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)  # Output shape: (None, 7, 7, 2048)

    skip_input = x  # for later skip connection

    # Feature Head 1 + Residual
    x1 = FeatureHead(2048, 512, name="feature_head_1")(x)
    x = ResidualBlock(512, name="res_block_1")(x1)

    # Feature Head 2 + Residual
    x2 = FeatureHead(512, 256, name="feature_head_2")(x)
    x = ResidualBlock(256, name="res_block_2")(x2)

    # Feature Head 3 + Residual
    x3 = FeatureHead(256, 128, name="feature_head_3")(x)
    x = ResidualBlock(128, name="res_block_3")(x3)

    # Skip connection from ResNet50 output
    pooled_skip = layers.Conv2D(128, kernel_size=1)(skip_input)
    x = layers.Add()([x, pooled_skip])

    # Global Pool + LoRA + Dense Head
    x = layers.GlobalAveragePooling2D()(x)
    x = LoRAAdapter(units=128, rank=lora_rank)(x)
    x = layers.Dense(256, activation='sigmoid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="AgriLeafNetLite")