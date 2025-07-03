# models/custom_layers.py

import tensorflow as tf
from tensorflow.keras import layers

class FeatureHead(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, name="feature_head"):
        super().__init__(name=name)
        self.upconv = layers.Conv2DTranspose(input_channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('sigmoid')

        self.downconv = layers.Conv2D(output_channels, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('sigmoid')

    def call(self, inputs, training=False):
        x = self.upconv(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.downconv(x)
        x = self.bn2(x, training=training)
        return self.act2(x)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name="residual_block"):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.norm1 = ScaleNorm()

        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.norm2 = ScaleNorm()

        self.final_act = layers.Activation('sigmoid')
        self.project = None  # Will be created dynamically

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.conv2.filters:
            self.project = layers.Conv2D(
                self.conv2.filters, kernel_size=1, padding='same', name=f"{self.name}_projection"
            )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.norm2(x)

        shortcut = inputs if self.project is None else self.project(inputs)
        x = layers.add([x, shortcut])
        return self.final_act(x)

class ScaleNorm(layers.Layer):
    def __init__(self, epsilon=1e-5, name="scale_norm"):
        super().__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.g = self.add_weight(
            name='scale', shape=(1,),
            initializer='ones', trainable=True
        )

    def call(self, inputs):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        return self.g * inputs / (norm + self.epsilon)